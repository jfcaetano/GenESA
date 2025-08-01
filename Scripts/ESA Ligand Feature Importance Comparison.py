import pandas as pd
from rdkit import Chem
import re
import os
from sklearn.ensemble import RandomForestRegressor

# Remove warnings
from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Define the list of CSV files

csv_files = [
    ('Filtered_RNN_20k.csv', 'RNN-20k.csv'),
    ('Filtered_Trans_20k.csv', 'Trans-20k.csv'),
    ('Filtered_Generated_SMILES-pur01-s1.csv', 'Trans-6M.csv'),
    ('Database_6.csv', 'ESA.csv')]

catalysts = ['VO(acac)2', 'VOSO4', 'VO(OiPr)3', 'All_Catalysts']
results = {}

# Loop over each file
for input_file, output_file in csv_files:
    data = pd.read_csv(input_file).fillna(0)

    # Determine the correct column names based on the file
    smiles_column = 'Lig-SMILES' if input_file == 'Database_6.csv' else 'generated_smiles'
    yield_column = 'Yield' if input_file == 'Database_6.csv' else 'Predicted_Yield'
    catalyst_column = 'Cat_Structure' if input_file == 'Database_6.csv' else 'Original_Cat_Structure'

    if 'is_valid' not in data.columns:
        data['is_valid'] = True

    file_results = {}

    # Process each catalyst
    for catalyst in catalysts:
        if catalyst == 'All_Catalysts':
            catalyst_data = data[(data[yield_column] > 0) & (data['is_valid'] == True)]
        else:
            catalyst_data = data[(data[yield_column] > 0) &
                                 (data['is_valid'] == True) &
                                 (data[catalyst_column] == catalyst)]

        ligand_descriptors = catalyst_data.filter(regex='_Lig$')

        X = ligand_descriptors
        y = catalyst_data[yield_column]

        # Fit Random Forest model from ESA Yield
        print("Fitting ligands & feature importance ...")
        rf = RandomForestRegressor(n_estimators=350, max_depth=40, random_state=47)
        rf.fit(X, y)

        feature_importances = pd.Series(rf.feature_importances_, index=ligand_descriptors.columns)
        top_features = feature_importances

        file_results[catalyst] = top_features

    # Store results for each file
    results[output_file] = pd.DataFrame(file_results)

# Combine results from all files
combined_results = pd.concat(results, axis=1)
combined_results.to_csv('combined_results-0.csv', header=True)

print("Combined results exported")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7 16:03:45 2024

@author: jfcaetano
"""

import pandas as pd


df = pd.read_csv('combined_results-0.csv')

# Fixed reference column
reference_columns = ['ESA_VO(acac)2']

# Suffixes to compute differences
suffixes = ['VO(acac)2', 'VOSO4', 'VO(OiPr)3', 'All_Catalysts']

# Groups of columns to calculate differences
groups = ['New_ESA', 'RNN_20k', 'Trans_20k', 'Trans_6M']

# Create difference columns
for suffix in suffixes:
    for group in groups:
        col_to_compare = f"{group}_{suffix}"
        ref_col = f"ESA_{suffix}"
        if col_to_compare in df.columns and ref_col in df.columns:
            diff_col = f"Diff_{col_to_compare}"
            df[diff_col] = df[col_to_compare] - df[ref_col]

# Save DataFrame
df.to_csv('difference_calculations.csv', index=False)

print("Difference Calculations Saved")


