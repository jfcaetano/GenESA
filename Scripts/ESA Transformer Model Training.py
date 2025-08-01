# Install libraries in Google Colab
!pip install rdkit-pypi torch tqdm

import os
import re
import numpy as np
import pandas as pd
import logging
from collections import Counter, OrderedDict
from typing import List
from dataclasses import dataclass
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from google.colab import drive
from tqdm.notebook import tqdm

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Mount Google Drive
drive.mount('/content/drive')

# Define input and output paths for Google Drive
save_dir = '/content/drive/MyDrive/model_run'
os.makedirs(save_dir, exist_ok=True)
input_file_path = '/content/drive/MyDrive/model_run/chemlb-20k-rdkit.csv'


log = logging.getLogger(__name__)

# Tokenizer and Vocabulary Classes
@dataclass(frozen=True)
class special_tokens:
    pad     = 'PAD'
    start   = 'START'
    end     = 'END'

class Vocabulary:
    pad = special_tokens.pad
    srt = special_tokens.start
    end = special_tokens.end

    def __init__(self, tokens: List=None, starting_id: int=0):
        self.token2id = OrderedDict()
        self.id2token = OrderedDict()
        self.lastidx  = starting_id - 1  # Adjusted index
        if tokens:
            for token in tokens:
                self.add(token)

    def __getitem__(self, token):
        return self.token2id[token]

    def get_token(self, id):
        return self.id2token[id]

    def add(self, token):
        if not isinstance(token, str):
            print(f"Token must be a string, but the type is {type(token)}")
            print('Do nothing.')
            return False
        if token in self.token2id:
            # Token already exists
            return False
        self._add(token)
        return True

    def add_tokens(self, tokens):
        return [self.add(token) for token in tokens]

    def __contains__(self, token_or_id):
        return token_or_id in self.token2id

    def __eq__(self, other_vocabulary):
        return self.token2id == other_vocabulary.token2id

    def __len__(self):
        return len(self.token2id)

    def encode(self, tokens):
        """
        Encodes a list of tokens to a list of token IDs.
        """
        token_ids = [self.token2id[token] for token in tokens]
        return token_ids

    def decode(self, token_ids):
        """
        Decodes a list of token IDs to tokens.
        """
        tokens = []
        for idx in token_ids:
            token = self.id2token[idx]
            if (token == self.end) or (token == self.pad):
                break
            if token == self.srt:
                continue
            tokens.append(token)
        return tokens

    def _add(self, token):
        if token not in self.token2id:
            self.lastidx +=1
            self.token2id[token] = self.lastidx
            self.id2token[self.lastidx] = token

class SmilesTokenizer:
    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]
    pad = special_tokens.pad
    srt = special_tokens.start
    end = special_tokens.end

    @staticmethod
    def tokenize_smiles(data, with_begin_and_end=True):
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = SmilesTokenizer.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens
        tokens = split_by(data, SmilesTokenizer.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = [special_tokens.start] + tokens + [special_tokens.end]
        return tokens

    @staticmethod
    def untokenize_smiles(tokens):
        smi = ""
        for token in tokens:
            if token in [special_tokens.pad, special_tokens.end]:
                break
            if token != special_tokens.start:
                smi += token
        return smi

# Helper functions
def count_frequency_tokens(pd_smiles: pd.Series):
    tokens = Counter()
    for smi in pd_smiles:
        tokens.update(SmilesTokenizer.tokenize_smiles(smi, with_begin_and_end=False))
    return dict(sorted(tokens.items(), key=lambda x: x[1], reverse=True))

def create_vocabulary(tokens: list):
    """
    Creates a vocabulary for a list of tokens.
    """
    vocabulary = Vocabulary()
    tokens = [str(tk) for tk in tokens]
    vocabulary.add_tokens([special_tokens.pad, special_tokens.start, special_tokens.end] + sorted(tokens))
    return vocabulary

# Loading Data and Initializing Model
data = pd.read_csv(input_file_path).fillna(0)
descriptor_columns = [col for col in data.columns if col.endswith('_Lig')]
descriptors = data[descriptor_columns].select_dtypes(include=[np.number]).values
smiles_list = data["washed_openeye_smiles"].values

# Create Dataset and DataLoader
df = pd.DataFrame({'descriptors': list(descriptors), 'smiles': smiles_list})
tokens_counter = Counter(token for smi in data['washed_openeye_smiles'] for token in SmilesTokenizer.tokenize_smiles(smi, with_begin_and_end=False))
vocabulary = create_vocabulary(list(tokens_counter.keys()))

# Extract special token IDs
pad_token_id = vocabulary.token2id[special_tokens.pad]
start_token_id = vocabulary.token2id[special_tokens.start]
end_token_id = vocabulary.token2id[special_tokens.end]
vocab_size = len(vocabulary)

# Dataset and DataLoader
class MoleculeDataset(Dataset):
    def __init__(self, df, tokenizer, vocabulary):
        self.descriptors = df['descriptors'].values
        self.smiles = df['smiles'].values
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        descriptor = torch.tensor(self.descriptors[idx], dtype=torch.float)
        smiles = self.smiles[idx]
        # Tokenize the SMILES string
        tokens = self.tokenizer.tokenize_smiles(smiles)
        # Convert tokens to IDs
        smiles_encoded = torch.tensor(self.vocabulary.encode(tokens), dtype=torch.long)
        return descriptor, smiles_encoded

def collate_fn(batch):
    descriptors, smiles = zip(*batch)
    descriptors = torch.stack(descriptors)
    smiles_padded = nn.utils.rnn.pad_sequence(
        smiles, batch_first=True, padding_value=pad_token_id
    )
    return descriptors, smiles_padded

# Create dataset and dataloader
dataset = MoleculeDataset(df, SmilesTokenizer, vocabulary)
batch_size = 47
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model Definition
import torch.nn.functional as F

class DescriptorToSMILESModel(nn.Module):
    def __init__(self, descriptor_size, vocab_size, embedding_dim, num_heads, num_layers, pad_token_id):
        super(DescriptorToSMILESModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.fc = nn.Linear(descriptor_size, embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embedding_dim, vocab_size)
        self.pad_token_id = pad_token_id
        self.embedding_dim = embedding_dim  # Store embedding_dim for later use

    def forward(self, descriptors, smiles_input):
        batch_size = descriptors.size(0)
        seq_length = smiles_input.size(1)

        # Map descriptors to memory
        memory = self.fc(descriptors)
        memory = memory.unsqueeze(0)

        # Embed smiles_input
        tgt = self.embedding(smiles_input).transpose(0, 1)

        # Compute positional encoding dynamically
        positional_encoding = self.create_positional_encoding(seq_length, self.embedding_dim).to(tgt.device)
        tgt = tgt + positional_encoding.unsqueeze(1)

        tgt_mask = self.generate_square_subsequent_mask(seq_length).to(tgt.device)
        tgt_key_padding_mask = (smiles_input == self.pad_token_id)

        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = output.transpose(0, 1)  # (batch_size, seq_length, embedding_dim)
        logits = self.output(output)  # (batch_size, seq_length, vocab_size)
        return logits

    def sample(self, descriptor, max_length=100, temperature=1.0):
        """
        Generates a SMILES sequence by sampling from the model’s output probabilities.
        Args:
            descriptor (torch.Tensor): Descriptor for a single molecule.
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Controls the randomness of sampling (higher means more diverse).
        """
        # Prepare initial inputs
        descriptor = descriptor.unsqueeze(0)  # Add batch dimension
        memory = self.fc(descriptor)  # Map descriptor to embedding space
        memory = memory.unsqueeze(0)  # Adjust for transformer dimension

        input_token = torch.tensor([start_token_id]).unsqueeze(0).to(descriptor.device)
        generated_sequence = [start_token_id]

        for _ in range(max_length):
            # Embed and add positional encoding
            embedded = self.embedding(input_token)
            positional_encoding = self.create_positional_encoding(embedded.size(0), self.embedding_dim).to(embedded.device)
            embedded = embedded + positional_encoding.unsqueeze(1)

            # Generate next token probabilities
            decoder_output = self.transformer_decoder(
                tgt=embedded,
                memory=memory
            )[-1]  # Get the last token’s output
            logits = self.output(decoder_output) / temperature  # Scale by temperature
            probabilities = F.softmax(logits, dim=-1)

            # Sample from the probability distribution
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            # Stop if end token is generated
            if next_token == end_token_id:
                break
            generated_sequence.append(next_token)
            input_token = torch.tensor([[next_token]], device=descriptor.device)

        return generated_sequence

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(torch.float)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

# Model Initialization
descriptor_length = descriptors.shape[1]

embedding_dim = 24
num_heads = 6
num_layers = 3
num_epochs = 100

model = DescriptorToSMILESModel(
    descriptor_size=descriptor_length,
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    pad_token_id=pad_token_id)


# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Early Stopping Threshold
desired_loss = 0.1

# Specify the directory in Google Drive to save the final model and vocabulary
save_dir = '/content/drive/MyDrive/model_run'
os.makedirs(save_dir, exist_ok=True)

# List to store epoch and loss values for tracking
epoch_loss_list = []
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    epoch_loss = 0
    for descriptors_batch, smiles_input in tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False):
        descriptors_batch, smiles_input = descriptors_batch.to(device), smiles_input.to(device)
        optimizer.zero_grad()
        logits = model(descriptors_batch, smiles_input)
        logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        targets = smiles_input[:, 1:].contiguous().view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    epoch_loss_list.append({'epoch': epoch + 1, 'loss': avg_loss})

    model_filename = os.path.join(save_dir, '02-Trn-descriptor_to_smiles_model.pth')
    vocab_filename = os.path.join(save_dir, '02-Trn-vocabulary.pth')
    torch.save(model.state_dict(), model_filename)
    torch.save(vocabulary, vocab_filename)
    print(f"Model saved at '{model_filename}', Vocabulary saved at '{vocab_filename}'")

    if avg_loss <= desired_loss:
        print(f"Desired loss reached. Training stopped at epoch {epoch+1}.")
        break

# Save the epoch-loss tracking data to a CSV file
loss_csv_path = os.path.join(save_dir, '02-Trn-training_loss_epochs.csv')
loss_df = pd.DataFrame(epoch_loss_list)
loss_df.to_csv(loss_csv_path, index=False)
print(f"Epoch and loss data saved to '{loss_csv_path}'")

print("Training completed and model saved.")
