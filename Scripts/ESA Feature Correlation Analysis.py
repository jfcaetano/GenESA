#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:18:30 2024

@author: jfcaetano
"""

import pandas as pd
from scipy.stats import spearmanr
from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

data = pd.read_csv('combined_results-0.csv', index_col=0)

# Separate columns based on prefixes
original_smiles = data.filter(regex='^ESA_')
rnn_smiles = data.filter(regex='^RNN_20k_')
trans_smiles = data.filter(regex='^Trans_20k_')
new_trans_smiles = data.filter(regex='^Trans_6M_')

# Remove prefixes for consistent indexing
original_smiles.columns = original_smiles.columns.str.replace('^ESA_', '', regex=True)
rnn_smiles.columns = rnn_smiles.columns.str.replace('^RNN_20k_', '', regex=True)
trans_smiles.columns = trans_smiles.columns.str.replace('^Trans_20k_', '', regex=True)
new_trans_smiles.columns = new_trans_smiles.columns.str.replace('^Trans_6M_', '', regex=True)

catalysts = ['All_Catalysts', 'VO(acac)2', 'VOSO4', 'VO(OiPr)3']

# Dictionary to store comparison results
comparison_results = []

for catalyst in catalysts:
    # Extract top features for the given catalyst from all datasets
    original_features = original_smiles[catalyst].dropna()
    new_features = new_smiles[catalyst].dropna()
    rnn_features = rnn_smiles[catalyst].dropna()
    trans_features = trans_smiles[catalyst].dropna()
    new_trans_features = new_trans_smiles[catalyst].dropna()  # New category

    def calculate_correlations_and_differences(ref_features, target_features):
        common_descriptors = ref_features.index.intersection(target_features.index)
        if not common_descriptors.empty:
            spearman_corr, _ = spearmanr(ref_features.loc[common_descriptors],
                                         target_features.loc[common_descriptors])
            score_diff = (ref_features - target_features).loc[common_descriptors].abs()
            avg_score_diff = score_diff.mean() if not score_diff.empty else None
        else:
            spearman_corr = None
            avg_score_diff = None
        return len(common_descriptors), spearman_corr, avg_score_diff

    num_common_new, spearman_corr_new, avg_score_diff_new = calculate_correlations_and_differences(original_features, new_features)
    num_common_rnn, spearman_corr_rnn, avg_score_diff_rnn = calculate_correlations_and_differences(original_features, rnn_features)
    num_common_trans, spearman_corr_trans, avg_score_diff_trans = calculate_correlations_and_differences(original_features, trans_features)
    num_common_new_trans, spearman_corr_new_trans, avg_score_diff_new_trans = calculate_correlations_and_differences(original_features, new_trans_features)

    # Convert correlation and average importance difference values to percentages
    def to_percentage(value):
        return value * 100 if value is not None else None

    comparison_results.append({
        'Catalyst': catalyst,
        'Num_Common_Descriptors_New': num_common_new,
        'Spearman_Correlation_New (%)': to_percentage(spearman_corr_new),
        'Avg_Importance_Diff_New (%)': to_percentage(avg_score_diff_new),
        'Num_Common_Descriptors_RNN': num_common_rnn,
        'Spearman_Correlation_RNN (%)': to_percentage(spearman_corr_rnn),
        'Avg_Importance_Diff_RNN (%)': to_percentage(avg_score_diff_rnn),
        'Num_Common_Descriptors_Trans': num_common_trans,
        'Spearman_Correlation_Trans (%)': to_percentage(spearman_corr_trans),
        'Avg_Importance_Diff_Trans (%)': to_percentage(avg_score_diff_trans),
        'Num_Common_Descriptors_NewTrans': num_common_new_trans,
        'Spearman_Correlation_NewTrans (%)': to_percentage(spearman_corr_new_trans),
        'Avg_Importance_Diff_NewTrans (%)': to_percentage(avg_score_diff_new_trans)
    })

# Convert the comparison results to a DataFrame
comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('feature_importance_comparison_new_rnn_trans_newtrans_percentage.csv', index=False)

print("Comparison correlation results exported")



import pandas as pd
from scipy.stats import spearmanr
from rdkit import RDLogger
import warnings
import re

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# Load the data
df = pd.read_csv('combined_results-0.csv', index_col=0)
feature_types = df['Feature_Type'].unique()

results = {}

# Prefixes to extract column groups
prefixes = ['RNN_20k_', 'Trans_20k_', 'Trans_6M_', 'ESA_']
pattern = re.compile(rf"^({'|'.join(re.escape(prefix) for prefix in prefixes)})(.+)$")

for feature in feature_types:
    feature_data = df[df['Feature_Type'] == feature]

    # Group columns by prefix and suffix
    column_groups = {prefix: {} for prefix in prefixes}
    for col in feature_data.columns:
        match = pattern.match(col)
        if match:
            prefix, suffix = match.groups()
            column_groups[prefix][suffix] = col

    feature_results = {}

    # Calculate Spearman correlation and standard deviation
    for suffix, new_col in column_groups['ESA_'].items():
        correlations = {}
        std_devs = {}

        std_devs['ESA_'] = feature_data[new_col].std() * 100

        if suffix in column_groups['RNN_20k_']:
            rnn_col = column_groups['RNN_20k_'][suffix]
            correlations['RNN_20k_'] = spearmanr(feature_data[new_col], feature_data[rnn_col])[0] * 100
            std_devs['RNN_20k_'] = feature_data[rnn_col].std() * 100
        else:
            print(f"Missing RNN column for suffix {suffix}")

        if suffix in column_groups['Trans_20k_']:
            trans_col = column_groups['Trans_20k_'][suffix]
            correlations['Trans_20k_'] = spearmanr(feature_data[new_col], feature_data[trans_col])[0] * 100
            std_devs['Trans_20k_'] = feature_data[trans_col].std() * 100
        else:
            print(f"Missing Trans_20k column for suffix {suffix}")

        if suffix in column_groups['Trans_6M_']:
            esa_col = column_groups['Trans_6M_'][suffix]
            correlations['Trans_6M_'] = spearmanr(feature_data[new_col], feature_data[esa_col])[0] * 100
            std_devs['Trans_6M_'] = feature_data[esa_col].std() * 100
        else:
            print(f"Missing Trans_2M column for suffix {suffix}")

        # Combine results
        if correlations:
            feature_results[suffix] = {**correlations, **{f"{key}_StdDev(%)": val for key, val in std_devs.items()}}

    if feature_results:
        results[feature] = feature_results

# Convert results into a DataFrame
if results:
    results_df = pd.DataFrame.from_dict(
        {(feature, suffix): results[feature][suffix]
         for feature in results.keys()
         for suffix in results[feature].keys()},
        orient='index'
    )

    # Ensure column names align with generated metrics
    results_df.columns = [
        'RNN_Corr(%)', 'Tr20k_Corr(%)', 'Tr6M_Corr(%)',
        'ESA_StdDev(%)', 'RNN_StdDev(%)', 'Tr20k_StdDev(%)', 'Tr6M_StdDev(%)',
    ]

    print(results_df)
    results_df.to_csv('spearman_correlations_with_stddev_extended.csv')
else:
    print("No correlations were calculated. Confirm that matching columns exist with the correct prefixes.")
