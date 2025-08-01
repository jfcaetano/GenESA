# Install RDKit and tqdm in Google Colab
!pip install rdkit-pypi tqdm

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import warnings
from google.colab import drive
from tqdm.notebook import tqdm

# Mount Google Drive
drive.mount('/content/drive')

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Define the list of descriptors to calculate
my_descriptors = []
for desc_name in dir(Descriptors):
    if desc_name in ['BalabanJ', 'BertzCT', 'TPSA']:
        my_descriptors.append(desc_name)
    elif desc_name[:3] == 'Chi':
        my_descriptors.append(desc_name)
    elif 'VSA' in desc_name:
        my_descriptors.append(desc_name)
    elif 'Kappa' in desc_name:
        my_descriptors.append(desc_name)
    elif desc_name[:1] == 'H':
        my_descriptors.append(desc_name)
    elif desc_name[:1] == 'N':
        my_descriptors.append(desc_name)
    elif desc_name[:1] == 'M':
        my_descriptors.append(desc_name)

# Add '_Lig' suffix to each descriptor name for the DataFrame columns
my_descriptors_lig = [desc + '_Lig' for desc in my_descriptors]

# Load the generated SMILES file from Google Drive
input_file_path = '/content/drive/MyDrive/model_run/chembl31.csv'
valid_df = pd.read_csv(input_file_path).fillna(0)

# Function to calculate selected descriptors
def calculate_descriptors(smiles, descriptors):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(descriptors)
    desc_values = []
    for desc in descriptors:
        desc_values.append(getattr(Descriptors, desc)(mol))
    return desc_values

# Calculate descriptors for each SMILES in the valid_df DataFrame with a progress bar
descriptor_data = []
for smiles in tqdm(valid_df['washed_openeye_smiles'], desc="Calculating Descriptors"):
    descriptor_data.append(calculate_descriptors(smiles, my_descriptors))

# Create a DataFrame with the calculated descriptors and add '_Lig' suffix to columns
descriptor_df = pd.DataFrame(descriptor_data, columns=my_descriptors_lig)
result_df = pd.concat([valid_df.reset_index(drop=True), descriptor_df], axis=1)

# Define the directory and output path for saving the results
save_dir = '/content/drive/MyDrive/model_run'
os.makedirs(save_dir, exist_ok=True)
output_file_path = os.path.join(save_dir, 'chemlb-20k-rdkit.csv')
result_df.to_csv(output_file_path, index=False)

print(f"Descriptors calculated and saved to '{output_file_path}'")
