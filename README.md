# GenESA - Inverse Ligand Design for ESA Reactions
Repository for the article "Inverse Ligand Design: A Generative Data-Driven Model for Optimizing Vanadyl-Based Epoxidation Catalysts"

This repository contains Python scripts used for generative and ML models for vanadyl-based catalyst ligand design for the epoxidation of small alkenes and alcohols (ESA). The scripts are designed for various stages of descriptor calculation, model training, hyperparameter tuning, SMILES generation and result evaluation. Below is an explanation of the key scripts and their roles.

<div align="center">
  <img width="1014" height="256" alt="Screenshot 2025-08-01 at 12 13 28" src="https://github.com/user-attachments/assets/484bb9dc-29e8-4fed-81fb-a2c8929d1bf8" />
</div>

## Scripts Overview

### 1. 'mol-conversion.py'
This script is designed for **molecular data processing and conversion**. It helps convert molecular structures in the dataset into formats that can be used by the machine learning models. Key functionalities include:
- Converting molecular data (e.g., SMILES strings) into descriptor formats suitable for model input.
- Handling molecule-based transformations, ensuring that molecular representations are standardized and compatible with the model.
- The output of this script is typically used as input features for subsequent model training and evaluation.

### 2. 'model_preliminary_run.py'
This script performs an **initial model evaluation** to provide baseline performance metrics. It runs a set of pre-defined machine learning models without hyperparameter tuning, giving insight into how well each model performs on the original dataset before further optimization steps. Functions include:
- Loading the dataset and preparing it for modeling.
- Running baseline models: Random Forest, Gradient Boosting, Support Vector Machines, Neural Networks.
- Outputting preliminary performance metrics (`R²`, MAE, RMSE) to help guide future model selection and optimization steps.

### 3. 'ESA_data_augmentation.py'
This script is used for **data augmentation** on the ESA-Vanadium dataset. It enriches the dataset by generating synthetic data points or applying transformations to existing data. The key functions of this script include:
- Handling missing values or filling gaps in the dataset.
- Augmenting the dataset to expand the available feature space, enabling more robust model training.
- Ensures that augmented data remains representative of the original chemical reaction conditions, focusing on maintaining chemical consistency.
  
### 4. 'ESA-hyperparameter-opt.py'
This script performs **hyperparameter optimization** for a Random Forest model using the `RandomizedSearchCV` function from `scikit-learn`. The primary objective is to optimize the model's performance by searching over a predefined range of hyperparameters. The script:
- Loads the ESA-Vanadium dataset, excluding certain irrelevant features.
- Conducts Random Forest regression, applying a hyperparameter search over parameters like the number of estimators, depth, and minimum samples split.
- Evaluates model performance using metrics such as `R²`, mean absolute error (MAE), and root mean squared error (RMSE) over multiple iterations.
- Saves the optimized model results and the best hyperparameters to a CSV file for further analysis.

### 5. 'ESA_model_opt_run.py'
This script focuses on the **execution and evaluation** of the optimized RF machine learning model on the augmented ESA-Vanadium dataset. It is responsible for:
- Collecting results from each model's training and testing phases.
- Recording performance metrics, including training and test `R²`, MAE, and RMSE for each model and iteration.
- Saving the evaluation metrics and model-specific results for further analysis or comparison.

### 6. 'partial_dependency_plots.py'
This script generates **partial dependence plots (PDPs)** to interpret the behavior of the trained models. PDPs show how each feature impacts the predicted outcome, allowing for better understanding of the model's decision-making process. The script:
- Loads the trained model and dataset.
- Generates plots that display the relationship between individual features and the model's predictions.
- Visualizes feature importance and dependencies, helping to uncover key drivers in the reaction outcomes.

### 7. 'weight_analysis.py'
The script records optimized RF model performance across 10 iterations with varying synthetic data weights. These weights, ranging from 0.1 to 0.9, adjust the contribution of synthetic reactions in the training data, simulating different levels of experimental noise.

### 8. 'top_15_fi_RF.py'
Script to run the optimized RF model using the top 15 features based on Feature Importance of model prediction using all catalysts.
- Returns a csv with the model's predictions.

### 9. 'shap_analysis.py'
Same framework of the **execution and evaluation** of the optimized RF machine learning model on the augmented ESA-Vanadium dataset, while also yielding the SHAP analysis for the "All Catalysts" case.
- Returns a csv with the model's SHAP values.
- Generates a SHAP plot that displays feature influence in the final model's predictions.

---

# ESA Vanadium Database Folder Overview

This folder contains files used for the development and analysis of machine learning models for predicting outcomes of epoxidation reactions.

## Folder Contents

### 1. **Mol Files Folder**
The `Mol Files` folder contains molecular structure files in `.mol` format. These files represent the chemical structures of various catalysts, ligands, and substrates used in the epoxidation reactions. They serve as input for molecular descriptors that are converted into feature vectors for machine learning models.

### 2. **Augmented_ESA_Vanadium_Database.csv**
This CSV file contains an **augmented version of the ESA Vanadium dataset**, which includes additional synthetic data points or transformations applied to the original dataset to enhance model performance. The augmentation process involves:
- Expanding the feature space with additional descriptors.
- Filling gaps in the dataset by generating plausible reaction conditions or molecular structures to enhance model robustness.
- Ensuring the augmented data is chemically valid and consistent with real-world reaction conditions.

### 3. **ESA-Vanadium-Database-v1.csv**
This CSV file represents the **first version of the ESA Vanadium dataset**, which includes the raw experimental data collected from various epoxidation reactions. The data primarily consists of:
- Descriptors related to the catalysts, ligands, substrates, solvents, and reaction conditions.
- Target variables like `Yield`, and other outcome measures of the reaction.

### 4. **ESA-Vanadium-Database-with-descriptors.csv**
This file is an **enhanced version** of the original dataset (`ESA-Vanadium-Database-v1.csv`), where molecular descriptors for each component (catalyst, substrate, ligand) have been added. These descriptors are used as features for machine learning models and were likely generated from the `.mol` files in the `Mol Files` folder.

---

# Results Folder Overview

The `Results` folder contains output file named ESA-Model-Aug-Full-Results.xlsx. The file amasses the outcomes of experiments conducted on the `ESA Vanadium Database` for predicting epoxidation reaction results such as reaction yield

## File: `ESA-Model-Aug-Full-Results.xlsx`

This Excel file is structured into three key sheets:

### 1. **Model Training Size Results**
This sheet contains the results of model performance across varying training sizes, helping to evaluate how much training data is required for the models to perform effectively. It typically records performance metrics for different training sizes across multiple iterations of model training and testing, as depicted in Figure 5 of the Manuscript.

#### Key Columns:
- **Training Size**: The proportion of the dataset used for training (e.g., 10%, 20%, 50%, 90%).
- **Iteration**: The specific iteration or run number (e.g., 1 through 10).
- **Model**: The type of machine learning model used (e.g., Random Forest, Gradient Boosting).
- **Train R²**: R-squared value on the training set, indicating how well the model fits the training data for a given training size.
- **Test R²**: R-squared value on the test set, measuring the model's performance on unseen data.
- **Test MAE**: Mean absolute error on the test set, which reflects the average difference between predicted and actual values.
- **Test RMSE**: Root mean squared error on the test set, a measure of prediction accuracy that accounts for large errors.

### 2. **Opt RF/GB Model Results - Cat Type**
This sheet summarizes the **optimized model results** for different catalyst types (`Cat Type`) using the optimized RF algorithm. The models have been trained using the optimal hyperparameters identified through a search process (e.g., using `RandomizedSearchCV`), and the results are segmented by catalyst structure. The ‘Cat_Structure’ column indicates what types of target catalysts are predicted. ‘Cat_Structure = All’ column presents the individual results for the Table 1 of the manuscript.

#### Key Columns:
- **Catalyst Type**: The specific catalyst structure used in the reaction (e.g., VO(acac)2, VOSO4).
- **Train R²**: R-squared value on the training set for the optimized model.
- **Test R²**: R-squared value on the test set, representing the model’s ability to generalize to unseen data for each catalyst type.
- **Test MAE**: Mean absolute error on the test set for each catalyst type.
- **Test RMSE**: Root mean squared error on the test set, reflecting the overall prediction error for each catalyst type.

### 3. **Feature Importance RF/GGB Opt Model**
This sheet provides an analysis of **feature importance** for the optimized models. Feature importance refers to the relative contribution of each feature (e.g., molecular descriptors, reaction conditions) to the model's predictions. The higher the importance score, the more significant that feature is in driving the model’s predictions. This data is presented in Figure 11 of the manuscript.

#### Key Columns:
- **Feature**: The name of the feature or descriptor used in the model (e.g., molecular descriptors like molecular weight, bond count, or reaction conditions like temperature, solvent type).
- **Type**: The type of feature, such as whether it's a molecular descriptor, reaction condition, or another variable.
- **Source**: The origin of the feature, indicating whether it comes from the catalyst, ligand, experimental data, etc.
- **VO(acac)2, VO(OiPr)3, VCl2(salen), VO(salen), VOSO4**: These columns represent the **feature importance scores** for each feature as they relate to the specific catalyst types. The scores indicate how influential each feature is for the model’s predictions for each particular catalyst.
- **All**: This column shows the **overall feature importance** across all catalysts, combining the importance scores into a general metric to represent the feature’s contribution to the model’s performance for the entire dataset.

### 4. **Top 15 FI Score**
This sheet includes ML model epoxidation yield prediction performance with optimized RF algorithm, using only the top 15 descriptors form Feature Importance of the optimized RF algorithm using all descriptors with the 'All Catalysts' scenario. Results are considered for all catalysts in bulk.

### 5. **RF Opt - DA Weight Test**
Displays error metrics for ESA yield prediction with varying synthetic data weights and variation of test R2 scores for ESA yield prediction with RF algorithm with different synthetic data weights. Figure 10 of the manuscript presents two charts for results visualization.

---
