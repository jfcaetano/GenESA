#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:41:47 2024

@author: jfcaetano
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import random
from tqdm import tqdm

# Load datasets
dataset = pd.read_csv('Augmented_ESA_Vanadium_Database.csv').fillna(0)
database_6 = pd.read_csv('Database_6.csv')
desired_columns = list(database_6.columns)

exclude_cols = [
    'Cat_Structure', 'Original_Lab', 'Original_Cat_Structure', 'Laboratory', 'Solvent',
    'Catalyst', 'Substrate', 'Ligand', 'Oxidant', 'EE', 'Yield', 'Configuration',
    'Cat-SMILES', 'Lig-SMILES', 'Sub-SMILES', 'Sol-SMILES', 'Yield_bin'
]

X_names = [x for x in dataset.columns if x not in exclude_cols]
y_name = 'Yield'
X = dataset[X_names]
y = dataset[y_name]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=350, max_depth=40, random_state=47)
model.fit(X_scaled, y)

strictly_positive_descriptors = [
    'Solv_opt_freq20', 'Solv_opt_freq25', 'Solv_diele_cst',
    'Temp_K', 'Substrate_quant_mmol', 'Catalyst_quant_mmol', 'Ligand_quant_mmol',
    'Oxidant_quant_mmol', 'Additive_quant_mL', 'Solution_vol_mL', 'Time_h'
]

non_negative_descriptors = [
    'Solv_Hbond_ac', 'Solv_Hbond_bs', 'Solv_surf_tens', 'Solv_aromcity', 'Solv_elect_halo'
]

def generate_alternatives(target_yield, original_cat_structure, num_alternatives=5, tolerance=2.0):
    subset = dataset[dataset['Original_Cat_Structure'] == original_cat_structure].copy()
    if subset.empty:
        print(f"No data found for Original_Cat_Structure: {original_cat_structure}")
        return None

    X_subset = subset[X_names]
    X_subset_scaled = scaler.transform(X_subset)

    alternatives = []
    attempts = 0
    max_attempts = 10000
    initial_tolerance = tolerance

    with tqdm(total=num_alternatives, desc=f"Generating for {original_cat_structure} at target yield {target_yield}") as pbar:
        while len(alternatives) < num_alternatives and attempts < max_attempts:
            random_index = random.randint(0, len(X_subset_scaled) - 1)
            sample_features = X_subset_scaled[random_index].copy()

            perturbation_strength = 0.1 if attempts < 500 else 0.05
            perturbation = np.random.normal(0, perturbation_strength, size=sample_features.shape)
            sample_features_perturbed = sample_features + perturbation

            predicted_yield = model.predict(sample_features_perturbed.reshape(1, -1))[0]

            if abs(predicted_yield - target_yield) <= tolerance:
                sample_features_original = scaler.inverse_transform(sample_features_perturbed.reshape(1, -1))[0]
                alternative = {feature_name: value for feature_name, value in zip(X_names, sample_features_original)}
                alternative['Predicted_Yield'] = predicted_yield
                alternative['Original_Cat_Structure'] = original_cat_structure

                if all(alternative[desc] > 0 for desc in strictly_positive_descriptors) and \
                   all(alternative[desc] >= 0 for desc in non_negative_descriptors):
                    alternatives.append(alternative)
                    pbar.update(1)

            attempts += 1

            # Adapt tolerance if struggling to meet criteria
            if attempts % 1000 == 0 and len(alternatives) < num_alternatives:
                tolerance = min(tolerance + 1.0, initial_tolerance * 2)

    if not alternatives:
        print(f"No alternatives found within tolerance for target yield {target_yield}")
        return None

    alternatives_df = pd.DataFrame(alternatives)

    for col in desired_columns:
        if col not in alternatives_df.columns:
            alternatives_df[col] = 0

    final_columns = desired_columns + ['Predicted_Yield', 'Original_Cat_Structure']
    alternatives_df = alternatives_df[final_columns]

    return alternatives_df

condition_pairs = [
    (0.0, 'VO(acac)2'), (10.0, 'VO(acac)2'), (20.0, 'VO(acac)2'), (30.0, 'VO(acac)2'), (40.0, 'VO(acac)2'),
    (50.0, 'VO(acac)2'), (60.0, 'VO(acac)2'), (70.0, 'VO(acac)2'), (80.0, 'VO(acac)2'), (90.0, 'VO(acac)2'),
    (100.0, 'VO(acac)2'), (0.0, 'VO(OiPr)3'), (10.0, 'VO(OiPr)3'), (20.0, 'VO(OiPr)3'), (30.0, 'VO(OiPr)3'),
    (40.0, 'VO(OiPr)3'), (50.0, 'VO(OiPr)3'), (60.0, 'VO(OiPr)3'), (70.0, 'VO(OiPr)3'), (80.0, 'VO(OiPr)3'),
    (90.0, 'VO(OiPr)3'), (100.0, 'VO(OiPr)3'), (0.0, 'VOSO4'), (10.0, 'VOSO4'), (20.0, 'VOSO4'),
    (30.0, 'VOSO4'), (40.0, 'VOSO4'), (50.0, 'VOSO4'), (60.0, 'VOSO4'), (70.0, 'VOSO4'), (80.0, 'VOSO4'),
    (90.0, 'VOSO4'), (100.0, 'VOSO4')
]

num_alternatives = 1000
tolerance = 10.0
all_alternatives = []

for target_yield, original_cat_structure in condition_pairs:
    print(f"Generating alternatives for target_yield={target_yield}, original_cat_structure={original_cat_structure}")
    alternatives_df = generate_alternatives(target_yield, original_cat_structure, num_alternatives, tolerance)
    if alternatives_df is not None:
        alternatives_df['Target_Yield'] = target_yield
        all_alternatives.append(alternatives_df)
    else:
        print(f"No alternatives generated for target_yield={target_yield}, original_cat_structure={original_cat_structure}")

if all_alternatives:
    combined_alternatives_df = pd.concat(all_alternatives, ignore_index=True)
    for col in desired_columns:
        if col not in combined_alternatives_df.columns:
            combined_alternatives_df[col] = 0
    final_columns = desired_columns + ['Predicted_Yield', 'Original_Cat_Structure', 'Target_Yield']
    combined_alternatives_df = combined_alternatives_df[final_columns]
    combined_alternatives_df.to_csv('Lig-Generated_Descriptors.csv', index=False)
    print("Alternatives exported")
