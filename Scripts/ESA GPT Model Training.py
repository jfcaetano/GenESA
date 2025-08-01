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
        token_ids = [self.token2id[token] for token in tokens]
        return token_ids

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
    vocabulary = Vocabulary()
    tokens = [str(tk) for tk in tokens]
    vocabulary.add_tokens([special_tokens.pad, special_tokens.start, special_tokens.end] + sorted(tokens))
    return vocabulary

# Data Loading and Preparation
data = pd.read_csv(input_file_path).fillna(0)
descriptor_columns = [col for col in data.columns if col.endswith('_Lig')]
descriptors = data[descriptor_columns].select_dtypes(include=[np.number]).values
smiles_list = data["washed_openeye_smiles"].values

df = pd.DataFrame({'descriptors': list(descriptors), 'smiles': smiles_list})
tokens_counter = Counter(token for smi in data['washed_openeye_smiles'] for token in SmilesTokenizer.tokenize_smiles(smi, with_begin_and_end=False))
vocabulary = create_vocabulary(list(tokens_counter.keys()))

pad_token_id = vocabulary.token2id[special_tokens.pad]
start_token_id = vocabulary.token2id[special_tokens.start]
end_token_id = vocabulary.token2id[special_tokens.end]
vocab_size = len(vocabulary)

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
        tokens = self.tokenizer.tokenize_smiles(smiles)
        smiles_encoded = torch.tensor(self.vocabulary.encode(tokens), dtype=torch.long)
        return descriptor, smiles_encoded

def collate_fn(batch):
    descriptors, smiles = zip(*batch)
    descriptors = torch.stack(descriptors)
    smiles_padded = nn.utils.rnn.pad_sequence(smiles, batch_first=True, padding_value=pad_token_id)
    return descriptors, smiles_padded

dataset = MoleculeDataset(df, SmilesTokenizer, vocabulary)
batch_size = 47
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

class VanillaGPTSMILES(nn.Module):
    def __init__(self, vocab_size, descriptor_size, nblocks=12, nheads=12, nembed=768,
                 pe_dropout=0.1, at_dropout=0.2, pad_token_id=None):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, nembed, padding_idx=pad_token_id)
        self.pos_embed = PositionalEncoding(nembed, pe_dropout)
        self.descriptor_proj = nn.Linear(descriptor_size, nembed)

        # Decoder blocks (GPT-style)
        self.blocks = nn.ModuleList([DecoderBlock(nembed, nheads, at_dropout, nembed) for _ in range(nblocks)])
        self.norm = nn.LayerNorm(nembed)
        self.output_layer = nn.Linear(nembed, vocab_size)
        self.pad_token_id = pad_token_id
        self.apply(self.init_weights_)

    @torch.no_grad()
    def init_weights_(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, descriptors, smiles_input):
        descriptors_proj = self.descriptor_proj(descriptors)
        smiles_embedded = self.tok_embed(smiles_input)  # Embed SMILES input
        smiles_embedded += self.pos_embed(smiles_embedded)  # Add positional encoding

        # Target and padding masks
        target_mask = make_masked_input(smiles_input.size(1)).to(smiles_input.device)
        pad_mask = make_padding_mask(smiles_input, self.pad_token_id).to(smiles_input.device)

        # Add descriptors to embeddings
        x = smiles_embedded + descriptors_proj.unsqueeze(1)

        # Forward through decoder blocks
        for block in self.blocks:
            x = block(x, target_mask, pad_mask)
        x = self.norm(x)
        logits = self.output_layer(x)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Precompute positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, nheads, dropout, mlp_dim):
        super().__init__()
        self.self_att = nn.MultiheadAttention(embed_dim, nheads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, target_mask, pad_mask):
        att_out, _ = self.self_att(x, x, x, key_padding_mask=pad_mask, attn_mask=target_mask)
        x = self.norm1(x + self.dropout1(att_out))
        mlp_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + self.dropout2(mlp_out))
        return x

def make_masked_input(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

def make_padding_mask(input, pad_token_id):
    if pad_token_id is None:
        return None
    mask = input == pad_token_id
    mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

# Model Initialization
descriptor_length = descriptors.shape[1]
embedding_dim = 36
num_heads = 6
num_layers = 3
num_epochs = 100

model = VanillaGPTSMILES(
    vocab_size=vocab_size,
    descriptor_size=descriptor_length,
    nblocks=num_layers,
    nheads=num_heads,
    nembed=embedding_dim,
    pad_token_id=pad_token_id
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
desired_loss = 0.1
save_dir = '/content/drive/MyDrive/model_run'
os.makedirs(save_dir, exist_ok=True)

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

    model_filename = os.path.join(save_dir, '03-GPT_decriptor_to_smiles_model.pth')
    vocab_filename = os.path.join(save_dir, '03-GPT-vocabulary.pth')
    torch.save(model.state_dict(), model_filename)
    torch.save(vocabulary, vocab_filename)
    print(f"Model saved at '{model_filename}', Vocabulary saved at '{vocab_filename}'")

    if avg_loss <= desired_loss:
        print(f"Desired loss reached. Training stopped at epoch {epoch+1}.")
        break

loss_csv_path = os.path.join(save_dir, '03-GPT-training_loss_epochs.csv')
loss_df = pd.DataFrame(epoch_loss_list)
loss_df.to_csv(loss_csv_path, index=False)
print(f"Epoch and loss data saved to '{loss_csv_path}'")

print("Training completed and model saved.")
