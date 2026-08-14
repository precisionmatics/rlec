# RLEC — RNA-Ligand Extended Connectivity Fingerprint

RLEC adapts the PLEC fingerprint (Wójcikowski et al., *Bioinformatics* 2019) to RNA-ligand systems. For each RNA–ligand contact pair, it pairs the Morgan-style chemical environments of the RNA atom and the ligand atom across increasing depths and hashes the pairs into a count vector.

## Installation

```bash
pip install rlec
```

## Quick start

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
])
# X: np.ndarray shape (n, 4096)
```

## Feature sets

| `feat_set` | RNA atom invariant |
|---|---|
| 0 | Basic ECFP (atomic num, charge, degree, aromaticity, ring) |
| 1 | + nucleotide type (A/U/G/C/T/modified) — **best** |
| 2 | + PBS group (Phosphate/Sugar/Base) |
| 3 | + nucleotide type + PBS group |

## Performance

Validated on 143 RNA–ligand complexes (LOOCV, LightGBM):

| Method | LOOCV r |
|---|---|
| Ligand-only ECFP | 0.562 |
| RNA-only ECFP | 0.594 |
| Element-pair FP | 0.578 |
| **RLEC feat1** | **0.710** |

95% bootstrap CI: [0.616, 0.790]. RLEC vs ligand-only: Δr = +0.148 (p < 0.0005).

## Requirements

- Python ≥ 3.9
- numpy, scipy, biopython, rdkit

## Citation

If you use RLEC, please cite:

> Stalin A. *RLEC: RNA-Ligand Extended Connectivity Fingerprint for binding affinity prediction.* (2026)

> Wójcikowski M, Kukiełka M, Stepniak-Konieczna M, Antosiewicz JM, Siedlecki P.
> Development of a protein–ligand extended connectivity (PLEC) fingerprint and its
> application for virtual screening. *Bioinformatics* 2019;35(8):1334–1341.
