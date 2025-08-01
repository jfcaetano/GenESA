# Install necessary libraries
!pip install rdkit-pypi torch pandas numpy tqdm

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from collections import OrderedDict
from typing import List
from rdkit import Chem
from rdkit import RDLogger
import warnings
from tqdm.notebook import tqdm

# Define file paths from Google Drive
model_filename = '/content/drive/MyDrive/model_run/01-RNN-descriptor_to_smiles_model.pth'
vocab_filename = '/content/drive/MyDrive/model_run/01-RNN-vocabulary.pth'
training_data_path = '/content/drive/MyDrive/model_run/chemlb-20k-rdkit.csv'
test_data_path = '/content/drive/MyDrive/model_run/Lig-Generated_Descriptors.csv'

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
vocabulary = torch.load(vocab_filename)

# Reconstruct special token IDs and vocab_size
pad_token_id = vocabulary.token2id[special_tokens.pad]
start_token_id = vocabulary.token2id[special_tokens.start]
end_token_id = vocabulary.token2id[special_tokens.end]
vocab_size = len(vocabulary)

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
                logits = self.output(output[:, -1, :])  # Get the last step's output
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Predict next token
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

# Load training data from Google Drive
training_data = pd.read_csv(training_data_path).fillna(0)
descriptor_columns = [col for col in training_data.columns if col.endswith('_Lig')]
descriptors_training = training_data[descriptor_columns].select_dtypes(include=[np.number]).values
descriptor_length = descriptors_training.shape[1]

embedding_dim = 276
hidden_dim = 148

# Initialize the model using this descriptor length
model = DescriptorToSMILESModel(
    descriptor_size=descriptor_length,
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    pad_token_id=pad_token_id
)

# Load model state_dict
model.load_state_dict(torch.load(model_filename), strict=False)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load test data from Google Drive
test_data = pd.read_csv(test_data_path).fillna(0)
yield_columns = test_data[['Target_Yield', 'Predicted_Yield']].copy()
test_data = test_data.drop(columns=['Target_Yield', 'Predicted_Yield'])
descriptors_test = test_data[descriptor_columns].select_dtypes(include=[np.number]).values

if descriptors_test.shape[1] != descriptor_length:
    raise ValueError(f"Expected descriptors of length {descriptor_length}, but got {descriptors_test.shape[1]}")

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

descriptors_tensor = torch.tensor(descriptors_test, dtype=torch.float).to(device)

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

# Save results to Google Drive
output_path = '/content/drive/MyDrive/model_run/01-RNN-Generated_SMILES.csv'
test_data.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
