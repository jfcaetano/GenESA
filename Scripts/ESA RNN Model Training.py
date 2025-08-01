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
from rdkit import Chem, RDLogger
from google.colab import drive
from tqdm.notebook import tqdm

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
drive.mount('/content/drive')


log = logging.getLogger(__name__)

# Define file paths in Google Drive
input_file_path = '/content/drive/MyDrive/model_run/chemlb-20k-rdkit.csv'
save_dir = '/content/drive/MyDrive/model_run/'
os.makedirs(save_dir, exist_ok=True)

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
        self.lastidx  = starting_id - 1
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

# Data Loading and Preparation
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
batch_size = 42
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model Definition remains unchanged
class DescriptorToSMILESModel(nn.Module):
    def __init__(self, descriptor_size, vocab_size, embedding_dim, hidden_dim, pad_token_id):
        super(DescriptorToSMILESModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.fc = nn.Linear(descriptor_size, hidden_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, descriptors, smiles_input=None, max_len=100, generate=False):
        hidden = self.fc(descriptors).unsqueeze(0)  # Initialize hidden state

        if generate:
            # Initialize input with the START token for each sequence in the batch
            input_token = torch.full((descriptors.size(0), 1), start_token_id, dtype=torch.long, device=descriptors.device)
            generated_sequence = []

            for _ in range(max_len):
                embedded = self.embedding(input_token)
                output, hidden = self.rnn(embedded, hidden)
                logits = self.output(output[:, -1, :])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated_sequence.append(next_token)

                input_token = next_token  # Feed next token as input in the next time step

                # Break if END token is generated
                if (next_token == end_token_id).all():
                    break

            generated_sequence = torch.cat(generated_sequence, dim=1)  # Concatenate generated tokens along the sequence
            return generated_sequence

        else:
            # Regular forward pass for training
            embedded = self.embedding(smiles_input)
            output, _ = self.rnn(embedded, hidden)
            logits = self.output(output)
            return logits


# Model Initialization
descriptor_length = descriptors.shape[1]
embedding_dim = 276
hidden_dim = 148
num_epochs = 100
desired_loss = 0.3

model = DescriptorToSMILESModel(
    descriptor_size=descriptor_length,
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    pad_token_id=pad_token_id
)

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Track loss for each epoch
epoch_loss_list = []

# Training Loop with Progress Bar and Early Stopping
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

    # Early stopping and save the model if threshold reached
    if avg_loss <= desired_loss:
        print(f"Desired loss reached. Training stopped at epoch {epoch+1}.")
        break

# Save the model and vocabulary to Google Drive
model_filename = os.path.join(save_dir, '01-RNN-descriptor_to_smiles_model.pth')
vocab_filename = os.path.join(save_dir, '01-RNN-vocabulary.pth')
torch.save(model.state_dict(), model_filename)
torch.save(vocabulary, vocab_filename)
print(f"Model saved at '{model_filename}', Vocabulary saved at '{vocab_filename}'")

# Save epoch-loss data to Google Drive
loss_csv_path = os.path.join(save_dir, '01-RNN-training_loss_epochs.csv')
loss_df = pd.DataFrame(epoch_loss_list)
loss_df.to_csv(loss_csv_path, index=False)
print(f"Epoch and loss data saved to '{loss_csv_path}'")
