# GenESA - Inverse Ligand Design for ESA Reactions
Repository for the article "Inverse Ligand Design: A Generative Data-Driven Model for Optimizing Vanadyl-Based Epoxidation Catalysts"

This repository contains Python scripts used for generative and ML models for vanadyl-based catalyst ligand design for the epoxidation of small alkenes and alcohols (ESA). The scripts are designed for various stages of descriptor calculation, model training, hyperparameter tuning, SMILES generation and result evaluation. Below is an explanation of the key scripts and their roles.


<div align="center">
  <img src="Supporting Information/GAbstract.png" alt="image" width="1014" height="256">
</div>


## Scripts Overview


### 1. ESA Descriptor Generator
This script generates optimized experimental descriptors targeting specific yields for vanadium-based catalytic reactions. It uses an ML perturbation approach to simulate plausible reaction conditions. The outputs are used to inform downstream ligand structure generation.
1)	Loading and preprocessing the ESA-Vanadium dataset.
2)	Training a RandomForestRegressor to predict reaction yields.
3)	Applying controlled random perturbations to descriptor vectors to generate new data points that match a target yield within a defined tolerance.
4)	Filtering generated alternatives to ensure chemical plausibility using predefined descriptor constraints (e.g., positive values for temperature, concentration, etc.).
5)	Iterating over a grid of catalyst structures and target yields to produce a comprehensive set of yield-sensitive descriptor alternatives.
6)	Saving the final dataset (Lig-Generated_Descriptors.csv) with predicted yields and conditions for each generated entry.

### 2. Descriptor Generator ChEMBL 31 20k
Code calculates RDKit-based molecular descriptors for a dataset of ligands generated from the 20k ChEMBL-derived model. Key functionalities include:
1)	Defining a curated list of RDKit molecular descriptors, including structural indices (e.g., Chi, Kappa), van der Waals surface area (VSA), and electronic properties (TPSA, EStateIndex).
2)	Computing descriptors for each molecule, while handling invalid SMILES by returning NaNs.
3)	Outputting the enriched dataset (chemlb-20k-rdkit.csv) to Google Drive for downstream model training, visualization, or ligand optimization.

### 3. Descriptor Generator Mcule Commercial Database 6M
Code calculates molecular descriptors for a large-scale commercial compound dataset from the Mcule Commercial Database (MCD), profiling purchasable molecules for use in generative model training. It includes a validation step to ensure SMILES quality.

1)	Defining a curated list of molecular descriptors from RDKit, including topological indices (e.g., Chi, Kappa), surface area (VSA), and physicochemical properties (TPSA, BalabanJ).
2)	Processing the dataset in chunks of 1 million SMILES, applying parallelized descriptor calculations using the swifter and dask libraries for efficiency.
3)	Appending descriptor columns to the original data and progressively writing the results to disk (purchase-rdkit.csv) to manage memory use during computation.
4)	Validating and cleaning the descriptor-annotated dataset by filtering only valid SMILES strings via RDKit parsing.

### 4. ESA RNN Model Training 
This script implements a recurrent neural network (RNN) model that learns to translate numerical reaction descriptors into valid SMILES strings from the ChEMBL-derived dataset. 
1)	Defining a tokenization scheme for SMILES strings using regular expressions and building a vocabulary with special tokens (START, END, PAD).
2)	Encoding SMILES strings into token sequences and pairing them with molecular descriptors for supervised learning.
3)	Creating a MoleculeDataset class and corresponding DataLoader with proper sequence padding and batching.
4)	Implementing a custom RNN model (DescriptorToSMILESModel) that takes descriptors as input and generates SMILES character-by-character using a GRU-based decoder.
5)	Training the model over multiple epochs using a cross-entropy loss function, with early stopping triggered by a target loss threshold.
6)	Saving the trained model (01-RNN-descriptor_to_smiles_model.pth) and vocabulary for future generation or fine-tuning.

### 5. ESA Transformer Model Training 
This code implements a Transformer-based architecture that maps numerical reaction descriptors to valid SMILES strings. It is used to traind on the ChEMBL 31 and MCD datasets.
1)	Loading the descriptor-enriched SMILES dataset (ChEMBL 31 or MCD).
2)	Building a custom SMILES tokenizer and vocabulary, including handling of special tokens (START, END, PAD) and rare patterns such as Cl, Br, and ring closures.
3)	Encoding descriptor-SMILES pairs for training, using PyTorch Dataset and DataLoader classes with sequence padding.
4)	Defining a Transformer decoder architecture, which maps learned descriptor embeddings to character-level SMILES sequences via attention layers and positional encodings.
5)	Supporting both training and autoregressive generation modes with temperature-controlled sampling for diverse SMILES outputs.
6)	Training the model using cross-entropy loss, with logging of per-epoch loss and early stopping.

### 6. ESA GPT Model Training 
Code implements a GPT-style Transformer model to translate numerical reaction descriptors into SMILES strings. It leverages multi-head self-attention and positional encoding to capture long-range dependencies in molecular sequences. 
1)	Loading descriptor-annotated SMILES data.
2)	Implementing a custom SMILES tokenizer and vocabulary builder, including support for special characters, ring structures, and positional information.
3)	Preparing descriptor–SMILES pairs in PyTorch-compatible Dataset and DataLoader objects with dynamic padding.
4)	Defining the VanillaGPTSMILES model, a Transformer-based architecture with: SMILES embeddings and sinusoidal positional encodings; Linear projection of descriptor vectors into token space; Stacked Transformer decoder blocks with masked attention
5)	Training the model using cross-entropy loss and Adam optimization, with logging of epoch-wise loss and early stopping when reaching a desired loss threshold.

### 7. ESA RNN Generative SMILES Model
This script performs SMILES generation from optimized reaction descriptors using a previously trained RNN model and evaluates chemical validity:
1)	Loading trained RNN model (01-RNN-descriptor_to_smiles_model.pth), its corresponding vocabulary (01-RNN-vocabulary.pth), and Test descriptors generated from prior model steps (Lig-Generated_Descriptors.csv)
2)	Reconstructing the SMILES tokenizer and vocabulary to support sequence decoding.
3)	Initializing and loading the trained RNN model, including architecture parameters (embedding size, hidden size).
4)	Feeding the test descriptor set through the model to generate SMILES strings using greedy decoding, one token at a time.
5)	Checking the chemical validity of generated SMILES using RDKit (MolFromSmiles).
6)	Summarizing generation performance, total outputs, number and percentage of valid structures.
7)	Saving the final annotated dataset, including generated SMILES and validity flags, to 01-RNN-Generated_SMILES.csv.

### 8. ESA Transformer Generative SMILES Model
This code uses a trained Transformer decoder model to generate SMILES strings from optimized ligand descriptors and assess their chemical validity. It includes:
1)	Loading trained Transformer model (ChEMBL or MCD), its vocabulary (ChEMBL or MCD), and the descriptor-based test set (Lig-Generated_Descriptors.csv)
2)	Reconstructing the tokenizer and vocabulary structure with special token mappings.
3)	Rebuilding the Transformer model with the original architecture parameters, including multi-head attention and positional encodings.
4)	Applying greedy decoding to generate SMILES strings from each descriptor vector using the .sample() method.
5)	Evaluating the chemical validity of generated structures using RDKit (Chem.MolFromSmiles).
6)	Summarizing performance statistics: number and percentage of valid vs. invalid SMILES.
7)	Saving the final output to 02-Trn-Generated_SMILES.csv.

### 9. ESA GPT Generative SMILES Model
This code generates SMILES strings from ligand descriptors using a trained GPT-style Transformer decoder and evaluates the chemical validity of the outputs.
1)	Loading the trained GPT model (03-GPT_decriptor_to_smiles_model.pth), associated vocabulary (03-GPT-vocabulary.pth) and descriptor test data (Lig-Generated_Descriptors.csv)
2)	Reconstructing the SMILES tokenizer and vocabulary, including positional token mapping and decoding functionality.
3)	Defining the Transformer decoder architecture with descriptor-to-embedding projection, positional encoding and stacked decoder blocks with masked multi-head self-attention.
4)	Generating SMILES using autoregressive decoding stopping upon reaching the end token.
5)	Evaluating chemical validity of each generated SMILES string using RDKit parsing.
6)	Summarizing generation outcomes in terms of valid/invalid structure counts and percentages.
7)	Exporting results, including generated SMILES, validity flags.

### 10. Generated SMILES Evaluation
Comprehensive evaluation of SMILES strings generated by multiple descriptor-to-structure models. It computes validity, uniqueness, novelty, and descriptor-based similarity against reference datasets.
1)	Loading SMILES from generated output files.
2)	Assessing chemical validity, uniqueness (non-redundant valid SMILES), and novelty against two reference libraries: ChEMBL31 or MCD and ESA experimental library.
3)	Computing RDKit descriptors for valid generated molecules using topological, electronic, and spatial properties (e.g., Chi, TPSA, VSA).
4)	Calculating cosine similarity between the generated molecule’s descriptors and their corresponding optimized descriptor targets.
5)	Aggregating results across four generated SMILES datasets.
6)	Summarizing evaluation metrics.

### 11. Ligand SMILES Filtering and Curation
This script performs systematic curation, canonicalization, and validation of Ligands SMILES strings generated via descriptor-to-structure pipelines. It applies a multi-step filtering protocol to ensure chemical plausibility, structural consistency, and coordination compatibility of candidate ligands for vanadium binding. 
1)	Canonicalizing all SMILES using RDKit to ensure standardized representations prior to analysis and filtering.
2)	Performing detailed molecular validation with RDKit, including: sanitization error checks, salence correctness evaluation, kekulization feasibility and atom count and charge assessment
3)	Filtering invalid molecules based on:Molecular size (minimum number of atoms), excessive character repetition (syntactic artifacts), presence of multiple disconnected structures, overall neutrality (formal charge = 0) and inclusion of vanadium-compatible donor motifs (via SMARTS-based substructure screening)
4)	Integrating RDKit-based red flag diagnostics into a structured DataFrame.
5)	De-duplicating filtered entries based on both SMILES and their associated categorical labels.

### 12. ESA Ligand Feature Importance Comparison
This script conducts comparative feature importance analysis across multiple SMILES-generated datasets by leveraging the optimized Random Forest regression models trained to predict catalytic reaction yields. It includes:
1)	Loading and preprocessing multiple curated SMILES datasets:
2)	Standardizing column references across datasets to identify.
3)	Subsetting data by four catalyst identities: VO(acac)2, VOSO4, VO(OiPr)3 and All_Catalysts (aggregated across all entries).
4)	Training a Random Forest model for each catalyst condition in each dataset to predict yield from ligand descriptors and compute descriptor feature importance values.
5)	Aggregating and exporting all feature importance scores in a structured format.
6)	Performing pairwise comparisons of feature importance against the ESA reference for each catalyst, calculating the difference in importance values: Δ(External – ESA).

### 13. ESA Feature Correlation Analysis
This script performs a comparative analysis of feature importance profiles across descriptor-to-structure generative models using Spearman rank correlation and standard deviation. 
1)	Loading precomputed feature importance scores, previously generated.
2)	Dataset normalization, grouping descriptor columns by dataset origin.
3)	Global correlation benchmarking across catalysts:
4)	For each catalyst (VO(acac)2, VOSO4, VO(OiPr)3 and All_Catalysts), the script computes:
5)	Spearman correlation coefficient (ρ × 100) between ESA and each generated dataset.
6)	Mean absolute difference in importance scores (Δ).
7)	Number of common descriptors used in each comparison.

### 14. ESA Ligand & Catalyst Compatibility
This script evaluates descriptor-level compatibility and chemical similarity between ligands and vanadium-based catalysts, using experimental data and generated ligands. It includes:

• Parsing descriptor-annotated datasets, dynamically matching descriptor columns for ligand–catalyst pairs (e.g., _Lig and _Cat suffixes), enabling scalable cross-comparison across matched chemical features.
• Compatibility scoring:
i) Computes a Gaussian-weighted similarity between ligand and catalyst descriptors.
ii) Supports alternative methods such as scaled difference and sigmoid-based comparison (modular implementation).
iii) Aggregates scores to produce a single compatibility metric for each pair.
• Structural similarity with cosine similarity between ligand descriptor vectors. It also calculates this independently of the compatibility score
• Final combined score: weighted combination: 70% descriptor compatibility + 30% cosine similarity
• Two application modes:
i) New ligands vs. experimental catalysts (cross-validation of generative pipeline)
ii) Known ligands vs. known catalysts (internal benchmarking of ESA dataset)

### 15. ESA Synthetic Accessibility Score Calculation
This script calculates the Synthetic Accessibility Score (SAS) for a set of ligands, based on their SMILES representations. It is also used to assess the SAS for the exisitng ESA ligands and the generated substrates, assessing synthetic feasibility via de novo molecular design.
• Loading ligand SMILES from a CSV file.
• Computing Morgan fingerprints for each molecule using RDKit, with a radius of 2 to capture local atom environments.
• Retrieving fragment-based synthetic accessibility scores from a precompiled fragment penalty table (fpscores.pkl.gz) and combining them with structure-based penalties.
• Structural complexity penalties include:
i) Atom count size penalty
ii) Stereochemical complexity (number of chiral centers)
iii) Spiro and bridgehead atom penalties
iv)Macrocyclic ring penalty
v) Fingerprint density penalty (related to fragment diversity vs size)

• Final SA Score Transformation: Raw scores are scaled and normalized to range from 1 (very easy to synthesize) to 10 (very difficult)
• Batch Processing and Export final results.

### 16. New Substrate Generation
Script generates substrate SMILES from descriptor vectors using a trained transformer decoder and evaluates structural validity and descriptor similarity (same decoded for ligand SMILES generation).

• Model Inference
 Loads pre-trained descriptor-to-SMILES model and vocabulary.
 Inputs substrate descriptors (_Sub).
 Generates SMILES via decoding from descriptor embeddings.

• SMILES Generation & Validation
Converts token sequences to SMILES strings.
Assesses chemical validity using RDKit.
Cleans and formats output SMILES.

• Similarity Analysis
Calculates RDKit descriptors for generated SMILES.
Computes cosine similarity to original descriptors.

---
