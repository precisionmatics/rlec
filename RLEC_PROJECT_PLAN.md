# RLEC: RNA-Ligand Extended Connectivity Fingerprint
## Complete Step-by-Step Development Plan

**Based on:** PLEC (Wójcikowski et al., Bioinformatics 35(8):1334–1341, 2019)
**Adaptation:** RNA instead of protein; RNA-specific atom features; RNA-ligand databases

---

## Project Overview

RLEC extends the PLEC concept to RNA-ligand systems:
- PLEC pairs ECFP environments of protein atoms + ligand atoms at contact (<4.5 Å)
- RLEC pairs ECFP-style environments of RNA atoms + ligand atoms at contact (<4.5 Å)
- RNA-specific atom features: nucleotide type, RNA moiety (base/sugar/phosphate), secondary structure context

---

## Step-by-Step Pipeline

### STEP 01 — Environment Setup and Dependencies
**Script:** `scripts/step01_setup_environment.py`
- Install all required packages: rdkit, oddt, biopython, numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm, requests, scipy
- Verify RDKit can parse RNA PDB structures
- Verify ODDT is installed and functional
- Save environment info to `logs/step01_environment.json`

### STEP 02 — Data Collection: RNA-Ligand Complexes from PDB
**Script:** `scripts/step02_collect_pdb_data.py`
- Query PDB for RNA-containing entries with small molecule ligands
- Filter: RNA chains + ligand (HETATM), resolution < 3.5 Å, experimental method X-ray/cryo-EM
- Download mmCIF/PDB files for all hits
- Save: `data/raw/pdb_rna_ligand_list.csv`

### STEP 03 — Binding Affinity Collection
**Script:** `scripts/step03_collect_affinity_data.py`
- Query BindingDB, ChEMBL, BioLiP, Binding MOAD for RNA-ligand Kd/Ki/IC50
- Convert all affinities to pKd (negative log scale), same as PDBbind
- Merge with PDB structure list from Step 02
- Save: `data/processed/rna_ligand_affinity.csv`

### STEP 04 — Structure Parsing and Validation
**Script:** `scripts/step04_parse_structures.py`
- Parse each PDB/mmCIF structure
- Extract RNA chains and ligand molecules
- Validate: RNA chain exists, ligand has valid SMILES, no missing heavy atoms
- Flag problematic structures (modified nucleotides, metal ions, covalent ligands)
- Save: `data/processed/valid_complexes.csv`, `logs/step04_validation.json`

### STEP 05 — RNA Atom Feature Engineering
**Script:** `scripts/step05_rna_atom_features.py`
- Define RNA-specific atom features (extends ECFP features):
  * Standard ECFP: atomic number, isotope, #heavy neighbors, #H, formal charge, ring, aromatic
  * RNA-specific: nucleotide type (A/U/G/C/modified), RNA moiety (base/sugar/phosphate), strand position
- Implement RNA atom environment hashing function
- Test on example structure
- Save: `data/processed/rna_atom_feature_schema.json`

### STEP 06 — Contact Detection (RNA–Ligand Interface)
**Script:** `scripts/step06_contact_detection.py`
- For each complex: find all RNA atom–ligand atom pairs within 4.5 Å (3D distance)
- Use KD-tree for efficient spatial search
- Exclude water molecules, ions (configurable)
- Output contact pairs per complex
- Save: `data/processed/contacts/` (one file per complex)

### STEP 07 — RLEC Fingerprint Construction (Core Algorithm)
**Script:** `scripts/step07_rlec_fingerprint.py`
- Implement RLEC algorithm (mirrors PLEC):
  ```
  rlec_bits = []
  for rna_atom, ligand_atom in contacts(rna, ligand, cutoff=4.5):
      rna_ecfp = ECFP_hashes(rna_atom, depth=5)   # RNA side: depth 5
      lig_ecfp = ECFP_hashes(ligand_atom, depth=1)  # Ligand side: depth 1
      for envs_pair in zip_longest(rna_ecfp, lig_ecfp):
          rlec_bits.append(hash(envs_pair))
  ```
- Fold raw bits to FP size (65536 = 2^16 default)
- Implement as sparse bit vector (count or binary)
- Compute RLEC for all valid complexes
- Save: `data/features/rlec_fingerprints.npz`

### STEP 08 — Fingerprint Parameter Analysis
**Script:** `scripts/step08_parameter_analysis.py`
- Test RNA depth: 1–6, Ligand depth: 1–6, FP sizes: 4096/16384/32768/65536
- Sparsity analysis: filter bits with variance < 0.01
- Generate saturation plot (Fig. equivalent of PLEC Fig. 2)
- Save: `results/parameter_analysis.csv`, `figures/saturation_plot.png`

### STEP 09 — Dataset Splitting (Train/Test)
**Script:** `scripts/step09_dataset_split.py`
- Split dataset avoiding RNA family leakage (by RNA family/Rfam ID, similar to PLEC's Uniprot-based split)
- Core set: 20% held out for final testing
- Training set: remaining 80%
- Save: `data/processed/train_ids.txt`, `data/processed/test_ids.txt`

### STEP 10 — ML Model Training: Linear (SGD)
**Script:** `scripts/step10_train_linear.py`
- Train SGDRegressor with Huber loss + Elasticnet penalty
- Hyperparameter grid search
- Evaluate on core test set: Pearson R, RMSE, SD
- Save model: `models/rlec_linear.pkl`
- Save results: `results/step10_linear_results.json`

### STEP 11 — ML Model Training: Random Forest
**Script:** `scripts/step11_train_rf.py`
- Train RandomForestRegressor (100–500 trees)
- Feature importance analysis
- Evaluate on core test set
- Save model: `models/rlec_rf.pkl`
- Save results: `results/step11_rf_results.json`

### STEP 12 — ML Model Training: Neural Network (MLP)
**Script:** `scripts/step12_train_nn.py`
- Train MLPRegressor: 3 hidden layers × 200 neurons, ReLU, L-BFGS-B
- Also test deeper architectures
- Evaluate on core test set
- Save model: `models/rlec_nn.pkl`
- Save results: `results/step12_nn_results.json`

### STEP 13 — Cross-Validation Stability Analysis
**Script:** `scripts/step13_cross_validation.py`
- 10-fold CV with RNA family-aware splitting
- Report Rp per fold for all 3 models
- Generate box plot (equivalent of PLEC Fig. 4)
- Save: `results/cv_results.csv`, `figures/cv_stability.png`

### STEP 14 — Depth Parameter Optimization
**Script:** `scripts/step14_depth_optimization.py`
- Train all 36 depth combinations (RNA depth 1–6 × ligand depth 1–6)
- For each: linear, RF, NN × 4 FP sizes = 432 models
- Plot Rp vs FP size colored by model type (equivalent of PLEC Fig. 3)
- Confirm optimal: RNA depth=5, ligand depth=1
- Save: `results/depth_optimization.csv`, `figures/depth_optimization.png`

### STEP 15 — Comparison with Baselines
**Script:** `scripts/step15_baseline_comparison.py`
- Compare RLEC against:
  * Ligand-only ECFP4 model (no RNA info)
  * SMILES-based descriptors (RDKit 2D)
  * Vina-style scoring (if applicable to RNA)
  * Random baseline
- Generate comparison bar chart (equivalent of PLEC Fig. 7)
- Save: `results/baseline_comparison.csv`, `figures/baseline_comparison.png`

### STEP 16 — Feature Interpretability Analysis
**Script:** `scripts/step16_interpretability.py`
- For linear model: extract top positive/negative weight features
- Trace each bit back to RNA moiety type + ligand substructure
- Generate heatmap of RNA moiety × ligand atom type interactions
- Save: `results/feature_interpretation.csv`, `figures/feature_heatmap.png`

### STEP 17 — Final Evaluation and Report
**Script:** `scripts/step17_final_evaluation.py`
- Final test set evaluation of best model
- Generate all publication-quality figures
- Write summary report: `results/RLEC_final_report.md`
- Save all metrics: `results/final_metrics.json`

---

## Data Flow Summary

```
PDB + BindingDB/ChEMBL
        ↓ Step 02-03
  RNA-ligand complexes + affinities
        ↓ Step 04
  Validated structures
        ↓ Step 05-06
  RNA atom features + contact pairs
        ↓ Step 07
  RLEC fingerprint vectors (65536 bits)
        ↓ Step 08-09
  Train/test split
        ↓ Step 10-12
  Trained models (linear/RF/NN)
        ↓ Step 13-16
  CV, optimization, interpretation
        ↓ Step 17
  Final report + figures
```

---

## Expected Outputs

| File | Description |
|------|-------------|
| `data/raw/` | Raw PDB files and affinity data |
| `data/processed/` | Cleaned, validated dataset |
| `data/features/rlec_fingerprints.npz` | All RLEC fingerprint vectors |
| `models/rlec_*.pkl` | Trained models |
| `results/final_metrics.json` | Pearson R, RMSE, SD for all models |
| `figures/` | All publication-quality plots |
| `logs/` | Step-by-step execution logs |
