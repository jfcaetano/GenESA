######################################


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:03:26 2024

@author: jfcaetano
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
import math
import pickle
import gzip
from tqdm import tqdm

from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Path to the fragment scores file
fscore_file_path = 'fpscores.pkl.gz'
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)

# Load fragment scores
def readFragmentScores(name=fscore_file_path):
    global _fscores
    with gzip.open(name, 'rb') as f:
        data = pickle.load(f)
    _fscores = {i[j]: float(i[0]) for i in data for j in range(1, len(i))}

_fscores = None
readFragmentScores()

def calculateScore(mol):
    if mol is None:
        return None

    fp = generator.GetSparseCountFingerprint(mol)
    scores = [float(_fscores.get(frag, -4)) * count for frag, count in fp.GetNonzeroElements().items()]
    score1 = sum(scores) / sum(fp.GetNonzeroElements().values()) if scores else 0

    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    nBridgeheads, nSpiro = rdMolDescriptors.CalcNumBridgeheadAtoms(mol), rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nMacrocycles = sum(1 for x in mol.GetRingInfo().AtomRings() if len(x) > 8)

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = math.log10(nMacrocycles + 1) if nMacrocycles > 0 else 0

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
    score3 = -0.5 * math.log10(len(fp.GetNonzeroElements()) / nAtoms) if nAtoms > len(fp.GetNonzeroElements()) else 0

    sascore = score1 + score2 + score3
    sascore = 1 + (10 - 1) * ((sascore + 4) / (2.5 + 4))
    return min(max(sascore, 1), 10)

def process_smiles(file_path):
    df = pd.read_csv(file_path)
    sa_scores = []
    for smiles in tqdm(df['Ligand_Generated'], desc="Calculating SA Scores"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            sa_scores.append(calculateScore(mol))
        else:
            sa_scores.append(None)  # or some other indicator of failure
    df['SA_Score_Ligand'] = sa_scores
    df.to_csv('processed_smiles_6M.csv', index=False)
    print("SA Scores added and saved to processed_smiles.csv.")

if __name__ == "__main__":
    file_path = 'ESA-Ligands-New.csv'  # Path to the CSV containing the SMILES strings
    process_smiles(file_path)
