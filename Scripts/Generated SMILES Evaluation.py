import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.spatial.distance import cosine
from rdkit import RDLogger
import warnings
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Define descriptor names and column naming convention
my_descriptors = [desc_name for desc_name in dir(Descriptors) if desc_name in ['BalabanJ', 'BertzCT', 'TPSA']
                  or desc_name[:3] == 'Chi' or 'VSA' in desc_name or 'Kappa' in desc_name or desc_name[:1] in 'HNM']
my_descriptors_lig = [desc + '_Lig' for desc in my_descriptors]  # Column names with '_Lig' suffix

def load_smiles(file_path, column_name):
    return pd.read_csv(file_path, usecols=[column_name])

# Validity check
def check_validity(smiles_series):
    valid = smiles_series.apply(lambda x: isinstance(x, str) and Chem.MolFromSmiles(x) is not None)
    return 100 * valid.mean()

# Uniqueness check for valid SMILES only
def check_uniqueness(smiles_series):
    # Filter for valid SMILES
    valid_smiles = smiles_series[smiles_series.apply(lambda x: isinstance(x, str) and Chem.MolFromSmiles(x) is not None)]
    unique_count = len(valid_smiles.drop_duplicates())
    total_count = len(valid_smiles)
    return 100 * (unique_count / total_count) if total_count > 0 else 0

# Novelty check
def check_novelty(target_smiles, reference_smiles):
    filtered_target_smiles = target_smiles[target_smiles.apply(lambda x: isinstance(x, str))]
    reference_set = set(reference_smiles.dropna())
    novel = filtered_target_smiles.apply(lambda x: x not in reference_set)
    return 100 * novel.mean()

# Descriptor calculation
def calculate_descriptors(smiles, descriptors):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return [getattr(Descriptors, desc)(mol) if mol else np.nan for desc in descriptors]

# Define paths for files
reference_file1_path = '31cv_chemlb31-rdkit.csv'
reference_file2_path = 'Database_6.csv'
reference_file1_smiles = load_smiles(reference_file1_path, 'canonical_smiles')
reference_file2_smiles = load_smiles(reference_file2_path, 'Lig-SMILES')

# List of target files
target_files = [
    '01-RNN-Generated_SMILES.csv',
    '02-Trn-Generated_SMILES.csv',
    '03-GPT-Generated_SMILES.csv',
    'Generated_SMILES-pur01.csv']

all_results = []
total_entries = sum([len(pd.read_csv(file_path).query('is_valid == True')) for file_path in target_files])

with tqdm(total=total_entries, desc="Processing SMILES entries") as pbar:
    for target_file_path in target_files:
        data = pd.read_csv(target_file_path)
        smiles_series = data['generated_smiles']
        valid_data = data[data['is_valid'] == True].reset_index(drop=True)

        # Validity, uniqueness, novelty calculations
        validity_percentage = check_validity(smiles_series)
        uniqueness_percentage = check_uniqueness(smiles_series)
        novelty_percentage_ref1 = check_novelty(smiles_series, reference_file1_smiles['canonical_smiles'])
        novelty_percentage_ref2 = check_novelty(smiles_series, reference_file2_smiles['Lig-SMILES'])

        # Descriptor and similarity calculations with COSINE
        descriptor_values = [calculate_descriptors(smiles, my_descriptors) for smiles in valid_data['generated_smiles']]
        calculated_descriptors_df = pd.DataFrame(descriptor_values, columns=my_descriptors)
        similarity_scores = []

        for i in range(len(valid_data)):
            original_values = pd.to_numeric(valid_data.loc[i, my_descriptors_lig], errors='coerce').values
            calculated_values = pd.to_numeric(calculated_descriptors_df.loc[i], errors='coerce').values

            if not np.isnan(original_values).any() and not np.isnan(calculated_values).any() and np.isfinite(original_values).all() and np.isfinite(calculated_values).all():
                similarity_score = 1 - cosine(original_values, calculated_values)
                similarity_scores.append(similarity_score * 100)

            pbar.update(1)

        avg_similarity = np.nanmean(similarity_scores)
        std_similarity = np.nanstd(similarity_scores)

        all_results.append({
            'File Name': target_file_path.split('/')[-1],
            'Validity Percentage': validity_percentage,
            'Uniqueness Percentage': uniqueness_percentage,
            'Novelty Percentage vs Chemlb': novelty_percentage_ref1,
            'Novelty Percentage vs ESA': novelty_percentage_ref2,
            'RDKit Average Similarity': avg_similarity,
            'RDKit Std Deviation Similarity': std_similarity
        })

consolidated_results_df = pd.DataFrame(all_results)
consolidated_csv_path = 'consolidated_full_results.csv'
consolidated_results_df.to_csv(consolidated_csv_path, index=False)

print(f"Consolidated results saved to '{consolidated_csv_path}'")
