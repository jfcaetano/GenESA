#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 10:23:36 2025

@author: jfcaetano
"""


from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity


INPUT_CSV = "Generated_SMILES-pur01.csv" 
OUTPUT_CSV = "ligand_similarity.csv"
SMILES_COL = "generated_smiles"
VALID_COL = "is_valid"

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

DEDUPLICATE_BY_CANONICAL_SMILES = True


def coerce_true(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}

def canonicalize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return mol, can

def morgan_fp(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)

def mean_and_median_pairwise_similarity(fps):
    n = len(fps)
    if n < 2:
        return float("nan"), float("nan"), 0

    sims_all = []
    sims_sum = 0.0
    n_pairs = n * (n - 1) // 2
    for i in tqdm(range(1, n), desc="Pairwise similarities", unit="row"):
        sims = BulkTanimotoSimilarity(fps[i], fps[:i])
        sims_sum += float(np.sum(sims))
        sims_all.extend(sims)

    mean_sim = sims_sum / n_pairs
    median_sim = float(np.median(np.array(sims_all))) if sims_all else float("nan")
    return mean_sim, median_sim, n_pairs



def main():
    inp = Path(INPUT_CSV)
    if not inp.exists():
        raise FileNotFoundError(f"Input CSV not found: {inp.resolve()}")

    df_all = pd.read_csv(inp)
    for col in (SMILES_COL, VALID_COL):
        if col not in df_all.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # 1) filter to valid rows
    df = df_all[df_all[VALID_COL].map(coerce_true)].copy()
    parsed = [canonicalize(s) for s in df[SMILES_COL]]
    df["mol"] = [p[0] for p in parsed]
    df["canonical_smiles"] = [p[1] for p in parsed]
    df = df[df["mol"].notna()].copy()

    dfu = df.drop_duplicates(subset="canonical_smiles").copy() if DEDUPLICATE_BY_CANONICAL_SMILES else df
    fps = [morgan_fp(m) for m in tqdm(dfu["mol"], desc="Computing Morgan fingerprints", unit="mol")]

    mean_sim, median_sim, n_pairs = mean_and_median_pairwise_similarity(fps)
    out = pd.DataFrame([{
        "input_file": str(inp.resolve()),
        "fingerprint": f"Morgan(r={MORGAN_RADIUS}, nBits={MORGAN_NBITS})",
        "deduplicated_by_canonical_smiles": bool(DEDUPLICATE_BY_CANONICAL_SMILES),
        "n_rows_input": int(len(df_all)),
        "n_valid_rows": int(len(df)),
        "n_unique_structures": int(len(dfu)),
        "n_pairs": int(n_pairs),
        "mean_pairwise_tanimoto_similarity": float(mean_sim),
        "median_pairwise_tanimoto_similarity": float(median_sim),
        "mean_pairwise_tanimoto_distance": float(1.0 - mean_sim) if not math.isnan(mean_sim) else float("nan"),
    }])

    outp = Path(OUTPUT_CSV)
    out.to_csv(outp, index=False)

    # 7) print primary number(s)
    print("\n=== Ligand Similarity Summary (Morgan/Tanimoto) ===")
    if math.isnan(mean_sim):
        print("Mean pairwise similarity: NaN (need ≥ 2 structures)")
    else:
        print(f"Mean pairwise similarity (primary): {mean_sim:.4f}")
        print(f"Median pairwise similarity       : {median_sim:.4f}")
        print(f"(For diversity, 1 - mean similarity = {1.0 - mean_sim:.4f})")
    print(f"Result written to: {outp.resolve()}")

if __name__ == "__main__":
    main()
