import pandas as pd
import re
import warnings
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import SanitizeFlags

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Paths
input_file = '00-Generated_SMILES-pur01.csv'
output_file = '01-Generated_SMILES-pur01.csv'

MAX_LENGTH = 200
MAX_REPEATS = 10

# Coordination-capable donor patterns
donor_smarts_patterns = {
    'alcohol/phenol': '[OX2H]',
    'carbonyl': '[CX3]=[OX1]',
    'carboxylic acid': 'C(=O)[OX1H0-,OX2H1]',
    'ether': '[OD2]([#6])[#6]',
    'amine': '[NX3;H2,H1]',
    'imine': '[NX2]=[CX3]',
    'aromatic nitrogen': '[nH0]',
    'thioether': '[#16X2]'
}
donor_mols = [(label, Chem.MolFromSmarts(smarts)) for label, smarts in donor_smarts_patterns.items()]

# RDKit red flag checker
def detect_rdkit_red_flags(smiles):
    result = {
        'is_valid': True,
        'sanitization_error': None,
        'valence_issue': False,
        'problem_atoms': [],
        'kekulization_failed': False,
        'kekulization_error': None
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result['is_valid'] = False
            result['sanitization_error'] = 'MolFromSmiles failed'
            return result

        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            explicit_valence = atom.GetExplicitValence()
            total_valence = atom.GetTotalValence()
            default_valence = Chem.GetPeriodicTable().GetDefaultValence(symbol)
            if explicit_valence > total_valence or explicit_valence > default_valence:
                result['valence_issue'] = True
                result['problem_atoms'].append({
                    'atom_idx': atom.GetIdx(),
                    'symbol': symbol,
                    'explicit_valence': explicit_valence,
                    'total_valence': total_valence,
                    'default_valence': default_valence
                })

        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception as e:
            result['kekulization_failed'] = True
            result['kekulization_error'] = str(e)

    except Exception as e:
        result['is_valid'] = False
        result['sanitization_error'] = str(e)

    return result

# Helper filters
def get_num_atoms(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumAtoms() if mol else None
    except:
        return None

def has_excessive_repeats(smiles):
    return bool(re.search(r'(.)\1{%d,}' % MAX_REPEATS, smiles)) if isinstance(smiles, str) else False

def is_single_structure(smiles):
    return isinstance(smiles, str) and '.' not in smiles

def is_neutral_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == 0 if mol else False

def has_vanadium_compatible_donor(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(mol.HasSubstructMatch(pattern) for _, pattern in donor_mols)

# Load data
df = pd.read_csv(input_file)
df['generated_smiles'] = df['generated_smiles'].astype(str)

# Rename possible duplicate 'is_valid' column from input
if list(df.columns).count('is_valid') > 1:
    idx = [i for i, c in enumerate(df.columns) if c == 'is_valid'][0]
    df.columns.values[idx] = 'first_is_valid'

# RDKit red flag analysis
df['rdkit_check'] = df['generated_smiles'].apply(detect_rdkit_red_flags)
df_check = pd.json_normalize(df['rdkit_check'])
df = pd.concat([df, df_check], axis=1).drop(columns=['rdkit_check'])

# Ensure numeric and boolean types
df['num_atoms'] = pd.to_numeric(df['generated_smiles'].apply(get_num_atoms), errors='coerce')
df = df.dropna(subset=['num_atoms'])
df['num_atoms'] = df['num_atoms'].astype(int)

for col in ['is_valid', 'valence_issue', 'kekulization_failed']:
    df[col] = df[col].apply(lambda x: str(x).strip().lower() == 'true')

# Debug output for datatypes
print("\nColumn dtypes before filtering:")
print(df.dtypes[['num_atoms', 'is_valid', 'valence_issue', 'kekulization_failed']])

# Save original count
rows_before = len(df)

# Apply all filters
filtered_df = df[
    (df['num_atoms'] > 1) &
    (df['is_valid']) &
    (~df['valence_issue']) &
    (~df['kekulization_failed']) &
    (df['generated_smiles'].astype(str).str.len() <= MAX_LENGTH) &
    (~df['generated_smiles'].apply(has_excessive_repeats)) &
    (df['generated_smiles'].apply(is_single_structure)) &
    (df['generated_smiles'].apply(is_neutral_molecule)) &
    (df['generated_smiles'].apply(has_vanadium_compatible_donor))
]

rows_after_filtering = len(filtered_df)

# Drop duplicates
pre_dedup_count = len(filtered_df)
filtered_df = filtered_df.drop_duplicates(subset=['generated_smiles', 'Original_Cat_Structure'])
duplicates_removed = pre_dedup_count - len(filtered_df)
rows_final = len(filtered_df)

# Save output
filtered_df.to_csv(output_file, index=False)

# Report
print(f"\n✅ Filtered data saved to: {output_file}")
