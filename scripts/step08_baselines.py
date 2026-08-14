"""
STEP 08 — Baseline Comparisons
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Runs fair comparisons against:
  1. Ligand-only ECFP (no RNA pairing) — Morgan FP on ligand alone
  2. RNA-only ECFP (no ligand pairing) — ECFP on RNA contact atoms only
  3. RLEC feat0 (basic ECFP, no RNA-specific features) vs feat3 (full)
  4. Interaction fingerprint baseline: count vector of contact element pairs

All use LightGBM with Optuna-tuned hyperparams (best model from step07d).
All use LOOCV pooled Pearson r (n=143) for honest, consistent evaluation.

Outputs:
  results/step08_baseline_results.json
  logs/step08_baselines.json
"""

import sys, os, json, logging, warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

ROOT        = Path(__file__).parent.parent
CONTACT_DIR = ROOT / "data" / "contacts"
FEAT_DIR    = ROOT / "data" / "features"
RES_DIR     = ROOT / "results"
LOG_DIR     = ROOT / "logs"
DATA_CSV    = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def pearson_metrics(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan, np.nan
    r, _ = stats.pearsonr(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return round(float(r), 4), round(rmse, 4)


def get_lgb_params():
    """Load Optuna-tuned LightGBM params from step07d results."""
    res_path = Path(__file__).parent.parent / "results" / "step07d_optuna_results.json"
    with open(res_path) as f:
        d = json.load(f)
    return d["results"]["lgb_optuna"]["best_params"]


def run_lgb_loocv(X, y, label):
    """LOOCV pooled Pearson r using LightGBM Optuna params."""
    params = get_lgb_params()
    loo    = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=4)
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    r, _  = stats.pearsonr(y, y_pred)
    rmse  = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    r, rmse = round(float(r), 4), round(rmse, 4)
    log.info(f"  {label}: LOOCV r={r:.4f}  RMSE={rmse:.4f}")
    return {"loocv_r": r, "loocv_rmse": rmse}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Ligand-only ECFP (Morgan radius=2, size=4096)
# ─────────────────────────────────────────────────────────────────────────────

def build_ligand_ecfp(complexes, fp_size=4096, radius=2):
    X = np.zeros((len(complexes), fp_size), dtype=np.float32)
    for i, cdata in enumerate(complexes):
        mol = cdata.get("rdkit_mol")
        if mol is None:
            continue
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size, bitInfo=bi)
        arr = np.zeros(fp_size, dtype=np.float32)
        fp.GetOnBits()
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        X[i] = arr
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: RNA-only contact atom ECFP
# ─────────────────────────────────────────────────────────────────────────────

def build_rna_ecfp(complexes, cutoff=6.0, fp_size=4096):
    """Count vector of RNA contact atom hash (depth-0 invariant) folded to fp_size."""
    X = np.zeros((len(complexes), fp_size), dtype=np.float32)
    for i, cdata in enumerate(complexes):
        contacts = cdata["contacts"].get(cutoff, [])
        graph    = cdata.get("rna_bond_graph", {})
        atoms    = {a["idx"]: a for a in cdata["rna_atoms"]}
        seen = set()
        for rna_idx, _, _ in contacts:
            if rna_idx in seen:
                continue
            seen.add(rna_idx)
            a = atoms.get(rna_idx)
            if a is None:
                continue
            h = hash((a["atomic_num"], a["formal_charge"],
                      a["n_heavy_nbrs"], a["is_aromatic"],
                      a["in_ring"], a["nuc_type"], a["pbs"])) & 0xFFFFFFFF
            X[i][h % fp_size] += 1
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: Element-pair contact fingerprint (simple interaction FP)
# ─────────────────────────────────────────────────────────────────────────────

def build_element_pair_fp(complexes, cutoff=6.0, fp_size=4096):
    """
    For each contact, hash (rna_element, lig_element) pair → fold.
    Mimics simple interaction fingerprint (fingeRNAt-like).
    """
    X = np.zeros((len(complexes), fp_size), dtype=np.float32)
    lig_atom_map = {}
    for i, cdata in enumerate(complexes):
        lig_atoms = {a["idx"]: a for a in cdata["lig_atoms"]}
        rna_atoms = {a["idx"]: a for a in cdata["rna_atoms"]}
        contacts  = cdata["contacts"].get(cutoff, [])
        for rna_idx, lig_idx, dist in contacts:
            ra = rna_atoms.get(rna_idx)
            la = lig_atoms.get(lig_idx)
            if ra is None or la is None:
                continue
            h = hash((ra["element"], la["element"],
                      ra.get("pbs", 0), ra.get("nuc_type", 0))) & 0xFFFFFFFF
            X[i][h % fp_size] += 1
    return X


# ─────────────────────────────────────────────────────────────────────────────
# RLEC feat0 vs feat3 ablation (load from saved npz)
# ─────────────────────────────────────────────────────────────────────────────

def load_rlec_npz(cutoff, rna_d, lig_d, fp_size, feat):
    c_str = f"{cutoff:.1f}"
    name  = f"rlec_c{c_str}_r{rna_d}_l{lig_d}_s{fp_size}_f{feat}.npz"
    path  = FEAT_DIR / name
    if not path.exists():
        return None, None, None
    data = np.load(path)
    return data["X"], data["y"], data["ids"]


def main():
    log.info("=" * 60)
    log.info("RLEC Step 08 — Baseline Comparisons")
    log.info("=" * 60)

    t_start = datetime.now()

    # ── Load complexes ──
    log.info("Loading complexes...")
    pkl_files = sorted(CONTACT_DIR.glob("*_contacts.pkl"))
    complexes = []
    failed = []
    for f in pkl_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        sdf_path = Path(f"/home/stalin/Desktop/PDFL-RNA/NA-L/{d['pdb_id']}/{d['pdb_id']}_ligand.sdf")
        mol = None
        if sdf_path.exists():
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
            mol = next((m for m in suppl if m is not None), None)
            if mol is None:
                suppl2 = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=False)
                mol = next((m for m in suppl2 if m is not None), None)
        if mol is None:
            failed.append(d["pdb_id"])
            continue
        d["rdkit_mol"] = mol
        complexes.append(d)

    log.info(f"Loaded: {len(complexes)}  Failed: {len(failed)}")

    # ── Subtype map ──
    df_meta = pd.read_csv(DATA_CSV)[["pdb", "subtype"]]
    subtype_map = dict(zip(df_meta["pdb"], df_meta["subtype"]))
    pdb_ids  = np.array([c["pdb_id"] for c in complexes])
    y        = np.array([c["pKd"] for c in complexes], dtype=np.float32)
    subtypes = np.array([subtype_map.get(p, "unknown") for p in pdb_ids])

    log.info(f"n={len(y)}  pKd range=[{y.min():.2f},{y.max():.2f}]")

    results = {}

    # ── Baseline 1: Ligand-only ECFP ──
    log.info("\n--- BASELINE 1: Ligand-only ECFP (Morgan r=2, fp=4096) ---")
    X_lig = build_ligand_ecfp(complexes, fp_size=4096, radius=2)
    log.info(f"  X_lig shape: {X_lig.shape}  nonzero rows: {(X_lig.sum(1)>0).sum()}")
    results["ligand_ecfp"] = run_lgb_loocv(X_lig, y, "ligand_ecfp")

    # ── Baseline 2: RNA-only ECFP ──
    log.info("\n--- BASELINE 2: RNA contact atoms only (fp=4096, cutoff=6.0) ---")
    X_rna = build_rna_ecfp(complexes, cutoff=6.0, fp_size=4096)
    log.info(f"  X_rna shape: {X_rna.shape}  nonzero rows: {(X_rna.sum(1)>0).sum()}")
    results["rna_ecfp"] = run_lgb_loocv(X_rna, y, "rna_ecfp")

    # ── Baseline 3: Element-pair FP ──
    log.info("\n--- BASELINE 3: Element-pair interaction FP (fp=4096, cutoff=6.0) ---")
    X_ep = build_element_pair_fp(complexes, cutoff=6.0, fp_size=4096)
    log.info(f"  X_ep shape: {X_ep.shape}  nonzero rows: {(X_ep.sum(1)>0).sum()}")
    results["element_pair"] = run_lgb_loocv(X_ep, y, "element_pair")

    # ── Ablation: RLEC feat0 (no RNA features) ──
    log.info("\n--- ABLATION: RLEC feat0 (basic ECFP, no RNA-specific features) ---")
    X_f0, y_f0, ids_f0 = load_rlec_npz(6.0, 6, 3, 4096, 0)
    if X_f0 is not None:
        results["rlec_feat0"] = run_lgb_loocv(X_f0, y_f0, "rlec_feat0")
    else:
        log.warning("RLEC feat0 npz not found")

    # ── Ablation: RLEC feat3 (full RNA features) ──
    log.info("\n--- ABLATION: RLEC feat3 (full RNA features: nuc_type + PBS) ---")
    X_f3, y_f3, ids_f3 = load_rlec_npz(6.0, 6, 3, 4096, 3)
    if X_f3 is not None:
        results["rlec_feat3"] = run_lgb_loocv(X_f3, y_f3, "rlec_feat3")
    else:
        log.warning("RLEC feat3 npz not found")

    # ── Ablation: RLEC feat1 (best from grid search) ──
    log.info("\n--- ABLATION: RLEC feat1 (nuc_type only — grid search best) ---")
    X_f1, y_f1, ids_f1 = load_rlec_npz(6.0, 6, 3, 4096, 1)
    if X_f1 is not None:
        results["rlec_feat1"] = run_lgb_loocv(X_f1, y_f1, "rlec_feat1")
    else:
        log.warning("RLEC feat1 npz not found")

    # ── Save ──
    elapsed = (datetime.now() - t_start).total_seconds()
    out = {"timestamp": t_start.isoformat(), "elapsed_sec": elapsed,
           "model": "LightGBM (Optuna-tuned)", "eval": "LOOCV pooled r (n=143)",
           "n_complexes": len(complexes), "results": results}

    def to_python(obj):
        if isinstance(obj, dict):   return {str(k): to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [to_python(v) for v in obj]
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    out_path = RES_DIR / "step08_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(to_python(out), f, indent=2)
    with open(LOG_DIR / "step08_baselines.json", "w") as f:
        json.dump(to_python(out), f, indent=2)

    # ── Final comparison table ──
    log.info(f"\n{'='*60}")
    log.info("COMPARISON TABLE (LOOCV pooled r, LightGBM Optuna, n=143)")
    log.info(f"{'='*60}")
    labels = {
        "ligand_ecfp" : "Ligand-only ECFP",
        "rna_ecfp"    : "RNA-only ECFP",
        "element_pair": "Element-pair FP",
        "rlec_feat0"  : "RLEC feat0 (basic, no RNA feats)",
        "rlec_feat1"  : "RLEC feat1 (+nuc_type)  ← BEST",
        "rlec_feat3"  : "RLEC feat3 (+nuc+PBS)",
    }
    for key, label in labels.items():
        if key not in results:
            continue
        s = results[key]
        log.info(f"  {label:<35}: r={s['loocv_r']:.4f}  RMSE={s['loocv_rmse']:.4f}")
    log.info(f"\nDone in {elapsed:.1f}s  |  Saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
