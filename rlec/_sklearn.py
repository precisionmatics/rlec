"""
rlec._sklearn — scikit-learn compatible transformer for RLECFingerprint.

Usage:
    from rlec import RLECTransformer
    from sklearn.pipeline import Pipeline
    from lightgbm import LGBMRegressor

    pipe = Pipeline([
        ("fp",  RLECTransformer(n_jobs=-1)),
        ("lgb", LGBMRegressor()),
    ])
    pipe.fit(complexes_train, y_train)
    pipe.predict(complexes_test)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from rlec.fingerprint import RLECFingerprint


class RLECTransformer(BaseEstimator, TransformerMixin):
    """scikit-learn Transformer wrapping RLECFingerprint.

    Accepts a list of (rna_pdb, ligand) tuples as X and produces an
    (n_samples, fp_size) float32 feature matrix. Fully compatible with
    sklearn Pipeline, GridSearchCV, and cross_val_score.

    Parameters
    ----------
    rna_depth : int, default 6
        Morgan iteration depth for RNA atoms.
    ligand_depth : int, default 3
        Morgan iteration depth for ligand atoms.
    fp_size : int, default 4096
        Folded fingerprint size.
    feat_set : int, default 1
        RNA feature set: 0=basic, 1=+nucleotide_type (best), 2=+PBS, 3=+both.
    cutoff : float, default 6.0
        Contact distance cutoff in Angstroms.
    n_jobs : int, default 1
        Parallel workers for batch computation. -1 uses all CPUs.
    show_progress : bool, default False
        Show tqdm progress bar during transform.

    Examples
    --------
    >>> from rlec import RLECTransformer
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> pipe = Pipeline([("fp", RLECTransformer()), ("rf", RandomForestRegressor())])
    >>> pipe.fit(train_complexes, y_train)   # train_complexes = [(rna.pdb, lig.sdf), ...]
    >>> pipe.predict(test_complexes)
    """

    def __init__(
        self,
        rna_depth: int = 6,
        ligand_depth: int = 3,
        fp_size: int = 4096,
        feat_set: int = 1,
        cutoff: float = 6.0,
        n_jobs: int = 1,
        show_progress: bool = False,
    ):
        self.rna_depth = rna_depth
        self.ligand_depth = ligand_depth
        self.fp_size = fp_size
        self.feat_set = feat_set
        self.cutoff = cutoff
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def fit(self, X, y=None) -> "RLECTransformer":
        """No-op — RLEC is a fixed featuriser with no trainable state."""
        return self

    def transform(self, X) -> np.ndarray:
        """Compute RLEC fingerprints for a list of (rna_pdb, ligand) pairs.

        Parameters
        ----------
        X : list of (rna_pdb, ligand) tuples
            rna_pdb — path to RNA PDB file
            ligand  — path to SDF file or RDKit Mol with 3D conformer

        Returns
        -------
        np.ndarray, shape (n_samples, fp_size), dtype float32
        """
        fp = RLECFingerprint(
            rna_depth=self.rna_depth,
            ligand_depth=self.ligand_depth,
            fp_size=self.fp_size,
            feat_set=self.feat_set,
            cutoff=self.cutoff,
        )
        return fp.transform_batch(X, n_jobs=self.n_jobs, show_progress=self.show_progress)
