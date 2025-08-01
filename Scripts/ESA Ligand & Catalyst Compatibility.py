#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:48:13 2024

@author: jfcaetano
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# File paths
database_csv = "Database_6.csv"
new_ligands_csv = "Filtered_Generated_SMILES-pur01-s1.csv"

# Load CSV files
database_df = pd.read_csv(database_csv)
new_ligands_df = pd.read_csv(new_ligands_csv)

# Column names for known data
ligand_smiles_col = "Lig-SMILES"
catalyst_smiles_col = "Cat-SMILES"
catalyst_name_col = "Cat_Structure"

# Column names for new data
new_ligand_smiles_col = "Ligand_SMILES"
new_catalyst_name_col = "Original_Cat_Structure"

# Filter for target catalysts
target_catalysts = ['VO(acac)2', 'VOSO4', 'VO(OiPr)3']
database_df = database_df[database_df[catalyst_name_col].isin(target_catalysts)]

# Identify descriptor columns dynamically
ligand_descriptor_cols = [col for col in database_df.columns if col.endswith("_Lig") and col != ligand_smiles_col]
catalyst_descriptor_cols = [col for col in database_df.columns if col.endswith("_Cat") and col != catalyst_smiles_col]

descriptor_pairs = [
    (ligand_col, catalyst_col)
    for ligand_col in ligand_descriptor_cols
    for catalyst_col in catalyst_descriptor_cols
    if ligand_col.replace("_Lig", "") == catalyst_col.replace("_Cat", "")
]

if not descriptor_pairs:
    raise ValueError("No matching ligand and catalyst descriptors found.")

# Function to calculate descriptor compatibility
def calculate_descriptor_compatibility(ligand_value, catalyst_value, method="gaussian", sigma=10):
    if pd.isnull(ligand_value) or pd.isnull(catalyst_value):
        return 0  # Incompatible if missing values
    diff = abs(ligand_value - catalyst_value)
    if method == "gaussian":
        return np.exp(-(diff ** 2) / (2 * sigma ** 2))  # Gaussian similarity
    elif method == "scaled_diff":
        return 1 / (1 + diff)  # Scaled difference
    elif method == "sigmoid":
        return 1 / (1 + np.exp(diff - sigma))  # Sigmoid similarity
    else:
        raise ValueError(f"Unknown method: {method}")

# Function to compute compatibility score
def calculate_compatibility(ligand_row, catalyst_row, method="gaussian", sigma=10):
    descriptor_scores = []
    for ligand_col, catalyst_col in descriptor_pairs:
        ligand_value = ligand_row[ligand_col]
        catalyst_value = catalyst_row[catalyst_col]
        score = calculate_descriptor_compatibility(ligand_value, catalyst_value, method=method, sigma=sigma)
        descriptor_scores.append(score)

    # Aggregate descriptor scores (average)
    if descriptor_scores:
        overall_score = np.mean(descriptor_scores)
    else:
        overall_score = 0
    return overall_score

# Function to compute compatibility and similarity grouped by catalyst types
def compute_compatibility_and_similarity(ligands_df, catalysts_df, ligand_column, catalyst_column, smiles_col, method="gaussian", sigma=10):
    results = []
    grouped_ligands = ligands_df.groupby(ligand_column)
    grouped_catalysts = catalysts_df.groupby(catalyst_column)

    for catalyst_name, ligand_group in grouped_ligands:
        if catalyst_name in grouped_catalysts.groups:
            catalyst_group = grouped_catalysts.get_group(catalyst_name)

            # Compute similarity
            similarity_matrix = cosine_similarity(
                ligand_group[ligand_descriptor_cols], catalyst_group[ligand_descriptor_cols]
            )

            for i, (_, ligand_row) in enumerate(ligand_group.iterrows()):
                for j, (_, catalyst_row) in enumerate(catalyst_group.iterrows()):
                    compatibility_score = calculate_compatibility(
                        ligand_row, catalyst_row, method=method, sigma=sigma
                    )
                    similarity_score = similarity_matrix[i, j]

                    combined_score = (0.7 * compatibility_score) + (0.3 * similarity_score)

                    results.append({
                        "Ligand_SMILES": ligand_row[smiles_col],
                        "Catalyst_SMILES": catalyst_row[catalyst_smiles_col],
                        "Catalyst_Name": catalyst_name,
                        "CompatibilityScore": compatibility_score,
                        "SimilarityScore": similarity_score,
                        "CombinedScore": combined_score,
                    })
    return pd.DataFrame(results)

# Step 1: Compute compatibility and similarity for new ligands by catalyst types
print("Computing compatibility and similarity for new ligands by catalyst types...")
new_compatibility_results = compute_compatibility_and_similarity(
    new_ligands_df, database_df, new_catalyst_name_col, catalyst_name_col, new_ligand_smiles_col, method="gaussian", sigma=10
)

# Remove duplicates by 'Ligand_SMILES' and 'Catalyst_SMILES'
new_compatibility_results = new_compatibility_results.drop_duplicates(subset=["Ligand_SMILES", "Catalyst_SMILES"])

# Save new ligand compatibility results
new_compatibility_results.to_csv("new_ligand_catalyst_compatibility_with_similarity.csv", index=False)
print("New ligand compatibility results saved to 'new_ligand_catalyst_compatibility_with_similarity.csv'.")

# Step 2: Compute compatibility and similarity for known ligands and catalysts
print("Computing compatibility and similarity for known ligands and catalysts...")
known_compatibility_results = compute_compatibility_and_similarity(
    database_df, database_df, catalyst_name_col, catalyst_name_col, ligand_smiles_col, method="gaussian", sigma=10
)

# Remove duplicates by 'Ligand_SMILES' and 'Catalyst_SMILES'
known_compatibility_results = known_compatibility_results.drop_duplicates(subset=["Ligand_SMILES", "Catalyst_SMILES"])

# Save known ligand compatibility results
known_compatibility_results.to_csv("known_ligand_catalyst_compatibility_with_similarity.csv", index=False)
print("Known ligand compatibility results saved to 'known_ligand_catalyst_compatibility_with_similarity.csv'.")
