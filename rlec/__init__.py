"""
RLEC — RNA-Ligand Extended Connectivity Fingerprint

Adapts the PLEC fingerprint (Wójcikowski et al., Bioinformatics 2019)
to RNA-ligand systems for binding affinity prediction.

Usage:
    from rlec import RLECFingerprint
    fp = RLECFingerprint(rna_depth=6, ligand_depth=3, fp_size=4096,
                         feat_set=1, cutoff=6.0)
    vec = fp.transform("rna.pdb", "ligand.sdf")

    # Physical interaction features (complements the fingerprint)
    from rlec import compute_physical_features
    phy = compute_physical_features("rna.pdb", "ligand.sdf")  # (29,) array

    # Combined (best for affinity prediction)
    import numpy as np
    combined = np.concatenate([vec, phy])                      # (4125,)

    # sklearn pipeline
    from rlec import RLECTransformer
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("fp", RLECTransformer(n_jobs=-1)), ("model", ...)])
"""

__version__ = "0.3.0"
__author__ = "Stalin A"
__email__ = "stalin.bioinfo@gmail.com"

from rlec.fingerprint import RLECFingerprint
from rlec._sklearn import RLECTransformer
from rlec.physical import (
    compute_physical_features,
    compute_physical_batch,
    PHYSICAL_FEATURE_NAMES,
)

__all__ = [
    "RLECFingerprint",
    "RLECTransformer",
    "compute_physical_features",
    "compute_physical_batch",
    "PHYSICAL_FEATURE_NAMES",
]
