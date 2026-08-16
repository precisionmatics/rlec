# RLEC: RNA-Ligand Extended Connectivity Fingerprint

[![PyPI version](https://img.shields.io/pypi/v/rlec.svg)](https://pypi.org/project/rlec/)
[![Python versions](https://img.shields.io/pypi/pyversions/rlec.svg)](https://pypi.org/project/rlec/)
[![PyPI downloads](https://img.shields.io/pypi/dm/rlec.svg)](https://pypi.org/project/rlec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/precisionmatics/rlec.svg?style=social)](https://github.com/precisionmatics/rlec)

RLEC adapts the PLEC fingerprint (Wójcikowski et al., *Bioinformatics* 2019) to RNA-ligand systems. For each RNA-ligand contact pair, it pairs the Morgan-style chemical environments of the RNA atom and the ligand atom across increasing depths and hashes the pairs into a count vector. A companion **physical interaction module** provides 29 physics-based descriptors that complement the fingerprint.

## Installation

```bash
pip install rlec
```

## Quick start

### RLEC Fingerprint

```python
from rlec import RLECFingerprint

fp = RLECFingerprint(
    rna_depth=6,      # Morgan depth for RNA atoms
    ligand_depth=3,   # Morgan depth for ligand atoms
    fp_size=4096,     # folded vector size
    feat_set=1,       # 1 = include nucleotide type (A/U/G/C) in RNA hash
    cutoff=6.0,       # contact distance cutoff (Å)
)

vec = fp.transform("path/to/rna.pdb", "path/to/ligand.sdf")
# vec: np.ndarray shape (4096,), dtype float32
```

`transform` also accepts an RDKit `Mol` object as the second argument (must have a 3D conformer).

For a batch:

```python
X = fp.transform_batch([
    ("rna1.pdb", "lig1.sdf"),
    ("rna2.pdb", "lig2.sdf"),
], n_jobs=-1)
# X: np.ndarray shape (n, 4096)
```

### Physical Interaction Features

29 physics-based descriptors computed directly from the 3D complex:

```python
from rlec import compute_physical_features, PHYSICAL_FEATURE_NAMES

phy = compute_physical_features("rna.pdb", "lig.sdf")
# phy: np.ndarray shape (29,), dtype float32
# Returns None if the PDB/SDF cannot be parsed

print(PHYSICAL_FEATURE_NAMES)  # list of 29 feature name strings
```

Features include: electrostatic energy (Gasteiger charges), H-bond count, hydrophobic contacts, ionic contacts (RNA phosphate ··· cationic ligand atom), π-stacking pairs, contact geometry statistics, and RDKit 2D ligand descriptors (MW, TPSA, HBD, HBA, rotatable bonds, rings).

For a batch:

```python
from rlec import compute_physical_batch

P = compute_physical_batch([
    ("rna1.pdb", "lig1.sdf"),
    ("rna2.pdb", "lig2.sdf"),
], n_jobs=-1)
# P: np.ndarray shape (n, 29)
```

### Combined descriptor (best for affinity prediction)

```python
import numpy as np
from rlec import RLECFingerprint, compute_physical_features

fp  = RLECFingerprint()
vec = fp.transform("rna.pdb", "lig.sdf")           # (4096,)
phy = compute_physical_features("rna.pdb", "lig.sdf")  # (29,)
combined = np.concatenate([vec, phy])               # (4125,)
```

### sklearn pipeline

```python
from rlec import RLECTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

pipe = Pipeline([
    ("fp", RLECTransformer(feat_set=1, n_jobs=-1)),
    ("model", GradientBoostingRegressor()),
])
pipe.fit(train_complexes, y_train)
```

## CLI

```bash
rlec info                             # show version, defaults, performance
rlec transform rna.pdb lig.sdf        # print fingerprint stats
rlec transform rna.pdb lig.sdf -o vec.npy   # save (4096,) vector
rlec physical  rna.pdb lig.sdf        # print all 29 physical features
rlec physical  rna.pdb lig.sdf -o feats.csv
rlec validate  rna.pdb lig.sdf        # inspect contacts, density
rlec batch data.csv --rna-col rna_path --lig-col lig_path -o feats.npz --n-jobs -1
```

## Feature sets

| `feat_set` | RNA atom invariant |
|---|---|
| 0 | Basic ECFP (atomic num, charge, degree, aromaticity, ring) |
| 1 | + nucleotide type (A/U/G/C/T/modified) **(recommended default)** |
| 2 | + PBS group (Phosphate/Sugar/Base) |
| 3 | + nucleotide type + PBS group |

## Performance

Validated on 143 RNA-ligand complexes from PDBbind NL2020.

**LOOCV (LightGBM), fingerprint quality:**

| Method | LOOCV r |
|---|---|
| Ligand-only ECFP | 0.562 |
| RNA-only ECFP | 0.594 |
| Element-pair FP | 0.578 |
| **RLEC feat1** | **0.710** |

95% bootstrap CI: [0.616, 0.790]. RLEC vs ligand-only: Δr = +0.148 (p < 0.0005).

**10-split 80/20 benchmark (LightGBM), same protocol as RLaffinity and Xia et al.:**

| Method | PCC | SPCC | RMSE | MAE |
|---|---|---|---|---|
| AutoDock Vina | -0.386 | -0.389 | 0.277 | 0.257 |
| RF-Score | 0.445 | 0.364 | 0.152 | 0.129 |
| RLaffinity | 0.559 | 0.540 | 0.152 | 0.119 |
| **RLEC + Physical** | **0.584** | **0.558** | **0.143** | **0.115** |
| RLASIF | 0.666 | 0.601 | 0.147 | 0.112 |

RLEC + Physical (4125-D combined) outperforms RLaffinity on all four metrics.

## Requirements

- Python ≥ 3.9
- numpy, scipy, biopython, rdkit, scikit-learn, pandas, tqdm, joblib

## Citation

If you use RLEC, please cite:

> Arulsamy S. *RLEC: an extended connectivity interaction fingerprint for RNA-ligand systems.* (2026). Available at: https://github.com/precisionmatics/rlec

> Wójcikowski M, Kukiełka M, Stepniak-Konieczna M, Antosiewicz JM, Siedlecki P.
> Development of a protein–ligand extended connectivity (PLEC) fingerprint and its
> application for virtual screening. *Bioinformatics* 2019;35(8):1334–1341.
