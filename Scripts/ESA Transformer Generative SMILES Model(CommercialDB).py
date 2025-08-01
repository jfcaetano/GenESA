# Install necessary libraries
!pip install numpy==1.24.4 pandas==1.5.3
!pip install rdkit-pypi torch tqdm


# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

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
import warnings

warnings.filterwarnings("ignore")

# Define file paths from Google Drive
vocab_filename = '/content/drive/MyDrive/LCA/Final_Model_Stats/purchase-vocabulary-chemb-10.pth'
model_filename = '/content/drive/MyDrive/LCA/Final_Model_Stats/purchase-descriptor_to_smiles_model-10.pth'
training_data_path = '/content/drive/MyDrive/LCA/purchase-filtered-cleaned.csv'
test_data_path = '/content/drive/MyDrive/model_run/Lig-Generated_Descriptors.csv'

# Load the original training data to get the correct descriptor column order
training_data_reference = pd.read_csv(training_data_path).fillna(0)
expected_descriptor_columns = [col for col in training_data_reference.columns if col.endswith('_Lig')]

# Suppress warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Define special_tokens class
@dataclass(frozen=True)
class special_tokens:
    pad     = 'PAD'
    start   = 'START'
    end     = 'END'

# Define Vocabulary class
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
            return False
        if token in self.token2id:
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
        return [self.token2id[token] for token in tokens]

    def decode(self, token_ids):
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

# Define SmilesTokenizer class
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

# Load the saved model and vocabulary
vocabulary = torch.load(vocab_filename, weights_only=False)

# Loading Data and Initializing Model
data = pd.read_csv(training_data_path).fillna(0)
descriptor_columns = [col for col in data.columns if col.endswith('_Lig')]

# Verify that descriptor columns in current training data match the reference CSV
assert descriptor_columns == expected_descriptor_columns, "Descriptor column order mismatch in training data!"


descriptors = data[descriptor_columns].select_dtypes(include=[np.number]).values
smiles_list = data["SMILES"].values

# Create Dataset and DataLoader
df = pd.DataFrame({'descriptors': list(descriptors), 'smiles': smiles_list})
tokens_counter = Counter(token for smi in data['SMILES'] for token in SmilesTokenizer.tokenize_smiles(smi, with_begin_and_end=False))

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
batch_size = 24
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
        memory = self.fc(descriptors)  # (batch_size, embedding_dim)
        memory = memory.unsqueeze(0)  # (1, batch_size, embedding_dim)

        # Embed smiles_input
        tgt = self.embedding(smiles_input).transpose(0, 1)  # (seq_length, batch_size, embedding_dim)

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
        )  # (seq_length, batch_size, embedding_dim)

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

# Model initialization
descriptor_length = descriptors.shape[1]
embedding_dim = 72
num_heads = 24
num_layers = 2
model = DescriptorToSMILESModel(
    descriptor_size=descriptor_length,
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    pad_token_id=pad_token_id
)

# Load model state_dict
model.load_state_dict(torch.load(model_filename), strict=False)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_data = pd.read_csv(test_data_path).fillna(0)
yield_columns = test_data[['Target_Yield', 'Predicted_Yield']].copy()
test_data = test_data.drop(columns=['Target_Yield', 'Predicted_Yield'])
descriptors_test = test_data[descriptor_columns].select_dtypes(include=[np.number]).values

if descriptors_test.shape[1] != descriptor_length:
    raise ValueError(f"Expected descriptors of length {descriptor_length}, but got {descriptors_test.shape[1]}")

descriptors_tensor = torch.tensor(descriptors_test, dtype=torch.float).to(device)

import torch

def generate_smiles(model, descriptors, tokenizer, vocabulary, device, max_length=100):
    model.eval()
    with torch.no_grad():
        # Prepare input token list starting with the start token
        input_tokens = [vocabulary.token2id[special_tokens.start]]
        input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, 1)

        for _ in range(max_length):
            # Forward pass: descriptors as memory, input_tensor as target input
            output_logits = model(descriptors, input_tensor)  # (1, seq_len, vocab_size)
            next_token_logits = output_logits[:, -1, :]  # Take logits for the last token
            next_token_id = next_token_logits.argmax(dim=-1).item()  # Greedy decoding

            # Append the next token to the input sequence
            input_tokens.append(next_token_id)
            input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)

            # Stop if the end token is generated
            if next_token_id == vocabulary.token2id[special_tokens.end]:
                break

        # Decode token IDs to SMILES string
        generated_smiles = tokenizer.untokenize_smiles(vocabulary.decode(input_tokens))
    return generated_smiles


generated_smiles_list, validity_list = [], []
for i in tqdm(range(descriptors_tensor.shape[0]), desc="Generating SMILES"):
    desc = descriptors_tensor[i].unsqueeze(0)
    smiles = generate_smiles(model, desc, SmilesTokenizer, vocabulary, device=device)
    generated_smiles_list.append(smiles)
    mol = Chem.MolFromSmiles(smiles)
    validity_list.append(mol is not None)

test_data['generated_smiles'] = generated_smiles_list
test_data['is_valid'] = validity_list
test_data[['Target_Yield', 'Predicted_Yield']] = yield_columns

valid_count = sum(validity_list)
invalid_count = len(validity_list) - valid_count
valid_percentage = (valid_count / len(validity_list)) * 100
invalid_percentage = (invalid_count / len(validity_list)) * 100

print(f"Total structures generated: {len(validity_list)}")
print(f"Valid structures: {valid_count} ({valid_percentage:.2f}%)")
print(f"Invalid structures: {invalid_count} ({invalid_percentage:.2f}%)")

# Save results
output_path = '/content/drive/MyDrive/LCA/Generated_SMILES-pur.csv'
test_data.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Sample DataFrame
df = pd.read_csv('/content/drive/MyDrive/LCA/Generated_SMILES-pur.csv')  # or however you're loading your data

# Remove everything after the first tab, space, or newline
df["generated_smiles"] = df["generated_smiles"].str.split().str[0]

# Save if needed
df.to_csv('/content/drive/MyDrive/LCA/Generated_SMILES-pur01.csv', index=False)

