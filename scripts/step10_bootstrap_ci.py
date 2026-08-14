"""
STEP 10 — Bootstrap Confidence Intervals + Statistical Significance
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Computes:
  1. Bootstrap 95% CI on Pearson r for RLEC best model (2000 iterations, LOOCV predictions)
  2. Bootstrap CI for feat0 (no RNA features) and feat3 (full RNA features) ablations
  3. Paired bootstrap comparison: RLEC vs ligand-only ECFP
     → H0: RNA pairing adds no predictive power (one-sided, p<0.05 to reject)
  4. Paired bootstrap comparison: RLEC feat1 vs feat0 (nucleotide-type effect)

All OOF predictions: LightGBM Optuna LOOCV (same protocol as step08 and step09).

Note on RSAPred benchmark (r=0.83): that result uses a single train/test split
on a different, smaller dataset (RNABSP) and is not directly comparable to our
LOOCV evaluation on n=143. We report it for context only.

Outputs:
  results/step10_bootstrap_results.json
  logs/step10_bootstrap.json
"""

import sys, os, json, logging, warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.model_selection import LeaveOneOut
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

ROOT        = Path(__file__).parent.parent
FEAT_DIR    = ROOT / "data" / "features"
CONTACT_DIR = ROOT / "data" / "contacts"
RES_DIR     = ROOT / "results"
LOG_DIR     = ROOT / "logs"
DATA_CSV    = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")

N_BOOTSTRAP  = 2000
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def bootstrap_pearson_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, alpha=0.05, seed=RANDOM_STATE):
    """Bootstrap 95% CI on Pearson r via percentile method."""
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        r, _ = stats.pearsonr(y_true[idx], y_pred[idx])
        boot_rs.append(float(r))
    boot_rs = np.array(boot_rs)
    lo = float(np.percentile(boot_rs, 100 * alpha / 2))
    hi = float(np.percentile(boot_rs, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4), round(float(np.mean(boot_rs)), 4), round(float(np.std(boot_rs)), 4)


def bootstrap_paired_comparison(y_true, y_pred_a, y_pred_b,
                                n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """
    Paired bootstrap test: is method A significantly better than method B?
    H0: delta_r = r(A) - r(B) <= 0
    p-value (one-sided): fraction of bootstrap samples where delta_r <= 0.
    95% CI on delta_r reported.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    r_a = float(stats.pearsonr(y_true, y_pred_a)[0])
    r_b = float(stats.pearsonr(y_true, y_pred_b)[0])
    delta_obs = r_a - r_b

    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        ra_b = float(stats.pearsonr(y_true[idx], y_pred_a[idx])[0])
        rb_b = float(stats.pearsonr(y_true[idx], y_pred_b[idx])[0])
        boot_deltas.append(ra_b - rb_b)

    boot_deltas = np.array(boot_deltas)
    lo  = float(np.percentile(boot_deltas, 2.5))
    hi  = float(np.percentile(boot_deltas, 97.5))
    p_val = float(np.mean(boot_deltas <= 0))

    return {
        "r_a"           : round(r_a, 4),
        "r_b"           : round(r_b, 4),
        "delta_r"       : round(delta_obs, 4),
        "delta_ci_lo"   : round(lo, 4),
        "delta_ci_hi"   : round(hi, 4),
        "p_value_onesided": round(p_val, 4),
        "significant_p05" : bool(p_val < 0.05),
    }


def get_lgb_params():
    with open(RES_DIR / "step07d_optuna_results.json") as f:
        return json.load(f)["results"]["lgb_optuna"]["best_params"]


def loocv_predict(X, y, label=""):
    """LOOCV OOF predictions with LightGBM Optuna params."""
    params = get_lgb_params()
    loo    = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=4)
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    r    = float(stats.pearsonr(y, y_pred)[0])
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    if label:
        log.info(f"  LOOCV {label}: r={r:.4f}  RMSE={rmse:.4f}")
    return y_pred, round(r, 4), round(rmse, 4)


def load_npz(cutoff, rna_d, lig_d, fp_size, feat):
    c_str = f"{cutoff:.1f}"
    name  = f"rlec_c{c_str}_r{rna_d}_l{lig_d}_s{fp_size}_f{feat}.npz"
    path  = FEAT_DIR / name
    if not path.exists():
        return None, None, None
    d = np.load(path)
    return d["X"], d["y"], d["ids"]


def build_ligand_ecfp(pdb_order, fp_size=4096, radius=2):
    """Ligand-only Morgan FP, ordered by pdb_order (must match RLEC ids)."""
    X = np.zeros((len(pdb_order), fp_size), dtype=np.float32)
    for i, pdb_id in enumerate(pdb_order):
        sdf_path = Path(f"/home/stalin/Desktop/PDFL-RNA/NA-L/{pdb_id}/{pdb_id}_ligand.sdf")
        if not sdf_path.exists():
            continue
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
        mol   = next((m for m in suppl if m is not None), None)
        if mol is None:
            suppl2 = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=False)
            mol    = next((m for m in suppl2 if m is not None), None)
        if mol is None:
            continue
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
        arr = np.zeros(fp_size, dtype=np.float32)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        X[i] = arr
    return X


def to_python(obj):
    if isinstance(obj, dict):           return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, bool):           return obj
    return obj


def main():
    log.info("=" * 60)
    log.info("RLEC Step 10 — Bootstrap CI + Statistical Significance")
    log.info("=" * 60)

    t_start = datetime.now()

    # ── Load best RLEC FP params ──
    df_res  = pd.read_csv(RES_DIR / "step05_grid_results.csv")
    rf_best = df_res[df_res["model"] == "rf"].dropna(subset=["pearson_r"]).nlargest(1, "pearson_r").iloc[0]
    cutoff  = float(rf_best["cutoff"])
    rna_d   = int(rf_best["rna_depth"])
    lig_d   = int(rf_best["lig_depth"])
    fp_size = int(rf_best["fp_size"])
    feat    = int(rf_best["feat_set"])
    log.info(f"Best FP params: cutoff={cutoff}  rna_d={rna_d}  lig_d={lig_d}  fp={fp_size}  feat={feat}")

    X, y, ids = load_npz(cutoff, rna_d, lig_d, fp_size, feat)
    log.info(f"RLEC FP shape: {X.shape}  n={len(y)}")

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: RLEC best — LOOCV + bootstrap CI
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- PART 1: RLEC best (feat{feat}) — LOOCV + {N_BOOTSTRAP}-iter bootstrap ---")
    y_rlec, r_rlec, rmse_rlec = loocv_predict(X, y, label=f"RLEC feat{feat}")
    lo, hi, mean_r, sd_r = bootstrap_pearson_ci(y, y_rlec)
    log.info(f"  95% CI = [{lo:.4f}, {hi:.4f}]  boot_mean={mean_r:.4f}")
    results["rlec_best"] = {
        "feat": feat, "loocv_r": r_rlec, "loocv_rmse": rmse_rlec,
        "ci_lo": lo, "ci_hi": hi, "boot_mean": mean_r, "boot_sd": sd_r,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: Ablation — feat0 (no RNA features) bootstrap CI
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- PART 2: feat0 ablation (no RNA features) ---")
    X0, y0, ids0 = load_npz(cutoff, rna_d, lig_d, fp_size, 0)
    if X0 is not None:
        y0_oof, r0, rmse0 = loocv_predict(X0, y0, label="RLEC feat0")
        lo0, hi0, m0, s0  = bootstrap_pearson_ci(y0, y0_oof)
        log.info(f"  95% CI = [{lo0:.4f}, {hi0:.4f}]")
        results["rlec_feat0"] = {
            "loocv_r": r0, "loocv_rmse": rmse0,
            "ci_lo": lo0, "ci_hi": hi0, "boot_mean": m0, "boot_sd": s0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PART 3: Ablation — feat3 (nuc_type + PBS) bootstrap CI
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- PART 3: feat3 ablation (nuc_type + PBS) ---")
    X3, y3, ids3 = load_npz(cutoff, rna_d, lig_d, fp_size, 3)
    if X3 is not None:
        y3_oof, r3, rmse3 = loocv_predict(X3, y3, label="RLEC feat3")
        lo3, hi3, m3, s3  = bootstrap_pearson_ci(y3, y3_oof)
        log.info(f"  95% CI = [{lo3:.4f}, {hi3:.4f}]")
        results["rlec_feat3"] = {
            "loocv_r": r3, "loocv_rmse": rmse3,
            "ci_lo": lo3, "ci_hi": hi3, "boot_mean": m3, "boot_sd": s3,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PART 4: Build ligand-only ECFP (same ordering as RLEC ids)
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- PART 4: Ligand-only ECFP baseline (same ordering as RLEC) ---")
    X_lig = build_ligand_ecfp(list(ids), fp_size=4096, radius=2)
    nonzero = int((X_lig.sum(1) > 0).sum())
    log.info(f"  X_lig shape: {X_lig.shape}  nonzero rows: {nonzero}")
    y_lig, r_lig, rmse_lig = loocv_predict(X_lig, y, label="ligand-only ECFP")
    lo_l, hi_l, m_l, s_l = bootstrap_pearson_ci(y, y_lig)
    log.info(f"  95% CI = [{lo_l:.4f}, {hi_l:.4f}]")
    results["ligand_only"] = {
        "loocv_r": r_lig, "loocv_rmse": rmse_lig,
        "ci_lo": lo_l, "ci_hi": hi_l, "boot_mean": m_l, "boot_sd": s_l,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # PART 5: Paired bootstrap — RLEC vs ligand-only
    # Tests whether RNA environment pairing adds significant predictive power
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- PART 5: Paired bootstrap: RLEC feat{feat} vs ligand-only ---")
    cmp_rlec_vs_lig = bootstrap_paired_comparison(y, y_rlec, y_lig)
    sig_str = "YES (p<0.05)" if cmp_rlec_vs_lig["significant_p05"] else "NO"
    log.info(f"  RLEC r={cmp_rlec_vs_lig['r_a']:.4f}  ligand-only r={cmp_rlec_vs_lig['r_b']:.4f}")
    log.info(f"  Δr={cmp_rlec_vs_lig['delta_r']:+.4f}  95%CI=[{cmp_rlec_vs_lig['delta_ci_lo']:.4f},{cmp_rlec_vs_lig['delta_ci_hi']:.4f}]")
    log.info(f"  p={cmp_rlec_vs_lig['p_value_onesided']:.4f}  Significantly better: {sig_str}")
    results["rlec_vs_ligand_only"] = cmp_rlec_vs_lig

    # ─────────────────────────────────────────────────────────────────────────
    # PART 6: Paired bootstrap — RLEC feat1 vs feat0 (nucleotide-type effect)
    # ─────────────────────────────────────────────────────────────────────────
    if X0 is not None and feat != 0:
        log.info(f"\n--- PART 6: Paired bootstrap: RLEC feat{feat} vs feat0 (RNA-feature effect) ---")
        cmp_feat = bootstrap_paired_comparison(y, y_rlec, y0_oof)
        sig_str2 = "YES (p<0.05)" if cmp_feat["significant_p05"] else "NO"
        log.info(f"  feat{feat} r={cmp_feat['r_a']:.4f}  feat0 r={cmp_feat['r_b']:.4f}")
        log.info(f"  Δr={cmp_feat['delta_r']:+.4f}  95%CI=[{cmp_feat['delta_ci_lo']:.4f},{cmp_feat['delta_ci_hi']:.4f}]")
        log.info(f"  p={cmp_feat['p_value_onesided']:.4f}  RNA features significant: {sig_str2}")
        results[f"feat{feat}_vs_feat0"] = cmp_feat

    # ─────────────────────────────────────────────────────────────────────────
    # Context note on RSAPred
    # ─────────────────────────────────────────────────────────────────────────
    results["rsapred_context"] = {
        "rsapred_r": 0.83,
        "rsapred_dataset": "RNABSP (smaller, different subset)",
        "rsapred_protocol": "single train/test split",
        "rlec_protocol": f"LOOCV n={len(y)}, pooled Pearson r",
        "note": (
            "RSAPred r=0.83 uses a different dataset and single-split evaluation; "
            "direct numerical comparison to RLEC LOOCV r is not valid. "
            "Both methods are included in Table 2 for context only."
        ),
    }

    # ── Save ──
    elapsed = (datetime.now() - t_start).total_seconds()
    out = {
        "timestamp"   : t_start.isoformat(),
        "elapsed_sec" : elapsed,
        "n_bootstrap" : N_BOOTSTRAP,
        "model"       : "LightGBM (Optuna-tuned)",
        "eval"        : f"LOOCV n={len(y)}, pooled Pearson r",
        "results"     : results,
    }

    out_path = RES_DIR / "step10_bootstrap_results.json"
    with open(out_path, "w") as f:
        json.dump(to_python(out), f, indent=2)
    with open(LOG_DIR / "step10_bootstrap.json", "w") as f:
        json.dump(to_python(out), f, indent=2)

    # ── Final summary ──
    log.info(f"\n{'='*60}")
    log.info("BOOTSTRAP SUMMARY")
    log.info(f"{'='*60}")
    for k in ["rlec_best", "rlec_feat0", "rlec_feat3", "ligand_only"]:
        if k not in results:
            continue
        v = results[k]
        log.info(f"  {k:<18}: LOOCV r={v['loocv_r']:.4f}  95%CI=[{v['ci_lo']:.4f},{v['ci_hi']:.4f}]")

    if "rlec_vs_ligand_only" in results:
        v = results["rlec_vs_ligand_only"]
        log.info(f"\n  RLEC vs ligand-only: Δr={v['delta_r']:+.4f}  "
                 f"CI=[{v['delta_ci_lo']:.4f},{v['delta_ci_hi']:.4f}]  "
                 f"p={v['p_value_onesided']:.4f}  significant={'YES' if v['significant_p05'] else 'NO'}")

    fk = f"feat{feat}_vs_feat0"
    if fk in results:
        v = results[fk]
        log.info(f"  RLEC feat{feat} vs feat0: Δr={v['delta_r']:+.4f}  "
                 f"p={v['p_value_onesided']:.4f}  significant={'YES' if v['significant_p05'] else 'NO'}")

    log.info(f"\nDone in {elapsed:.1f}s  |  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
