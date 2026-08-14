"""
RLEC — RNA-Ligand Extended Connectivity Fingerprint

Adapts the PLEC fingerprint (Wójcikowski et al., Bioinformatics 2019)
to RNA-ligand systems for binding affinity prediction.

Usage:
    from rlec import RLECFingerprint
    fp = RLECFingerprint(rna_depth=6, ligand_depth=3, fp_size=4096,
                         feat_set=1, cutoff=6.0)
    vec = fp.transform("rna.pdb", "ligand.sdf")

    # sklearn pipeline
    from rlec import RLECTransformer
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("fp", RLECTransformer(n_jobs=-1)), ("model", ...)])
"""

__version__ = "0.2.0"
__author__ = "Stalin A"
__email__ = "stalin.bioinfo@gmail.com"

from rlec.fingerprint import RLECFingerprint
from rlec._sklearn import RLECTransformer

__all__ = ["RLECFingerprint", "RLECTransformer"]
