
!pip install rdkit-pypi tqdm swifter dask numba

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import warnings
from google.colab import drive
from tqdm.notebook import tqdm
import swifter

# Mount Google Drive
drive.mount('/content/drive')

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Define the list of descriptors to calculate (filter only callable ones)
descriptor_prefixes = ['BalabanJ', 'BertzCT', 'TPSA', 'Chi', 'VSA', 'Kappa', 'H', 'N', 'M']
my_descriptors = [
    desc for desc in dir(Descriptors)
    if any(desc.startswith(prefix) or prefix in desc for prefix in descriptor_prefixes)
    and callable(getattr(Descriptors, desc))  # Ensure descriptor is a function
]

# Add '_Lig' suffix to each descriptor name for DataFrame columns
my_descriptors_lig = [desc + '_Lig' for desc in my_descriptors]

# Load the generated SMILES file from Google Drive
input_file_path = '/content/drive/MyDrive/LCA/purchase.csv'
chunk_size = 1_000_000  # Process in chunks of 1 million rows

# Define the directory and output path for saving results
save_dir = '/content/drive/MyDrive/LCA'
os.makedirs(save_dir, exist_ok=True)
output_file_path = os.path.join(save_dir, 'purchase-rdkit.csv')

# Function to calculate selected descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(my_descriptors)  # Return NaN for invalid molecules
    return [getattr(Descriptors, desc)(mol) if hasattr(Descriptors, desc) else np.nan for desc in my_descriptors]

# Process CSV in chunks and save results progressively
first_chunk = True
for chunk in tqdm(pd.read_csv(input_file_path, chunksize=chunk_size), desc="Processing Chunks"):
    chunk = chunk.rename(columns={chunk.columns[0]: "SMILES"})

    # Compute descriptors
    chunk[my_descriptors_lig] = chunk['SMILES'].swifter.apply(calculate_descriptors).apply(pd.Series)

    # Save chunk to CSV
    chunk.to_csv(output_file_path, mode='a', header=first_chunk, index=False)
    first_chunk = False  # Ensure headers are written only for the first chunk

print(f"Descriptors calculated and saved to '{output_file_path}'")



######################################

# Check Smiles Validity

# File paths
input_file_path = '/content/drive/MyDrive/LCA/purchase-rdkit.csv'
cleaned_output_file_path = '/content/drive/MyDrive/LCA/purchase-rdkit-cleaned.csv'
chunk_size = 1_000_000  # Process in chunks of 1 million rows

# Function to check and filter valid SMILES
def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

# Process CSV in chunks and save only valid SMILES
first_chunk = True
for chunk in tqdm(pd.read_csv(input_file_path, chunksize=chunk_size), desc="Filtering Invalid SMILES"):
    # Ensure correct column name
    chunk = chunk.rename(columns={chunk.columns[0]: "SMILES"})

    # Filter valid SMILES efficiently using swifter
    chunk = chunk[chunk['SMILES'].swifter.apply(is_valid_smiles)]

    # Save cleaned chunk to CSV
    chunk.to_csv(cleaned_output_file_path, mode='a', header=first_chunk, index=False)
    first_chunk = False  # Ensure headers are written only for the first chunk

print(f"✅ Cleaned file saved to '{cleaned_output_file_path}'")
