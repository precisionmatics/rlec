"""
STEP 09 — Per-Subtype Performance + Leave-One-Subtype-Out (LOSO)
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Two analyses:
  1. Per-subtype performance: train on 80% all data, report r on each subtype's test samples
  2. LOSO-CV: for each subtype, train on all other subtypes, test on that subtype
     → tests generalization across RNA families

Uses best params from step07 (or auto-selects from step05 results).

Outputs:
  results/step09_subtype_results.json
  logs/step09_subtype.json
"""

import sys, os, json, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / "data" / "features"
RES_DIR  = ROOT / "results"
LOG_DIR  = ROOT / "logs"
DATA_CSV = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")

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


def main():
    log.info("=" * 60)
    log.info("RLEC Step 09 — Per-Subtype + LOSO Analysis")
    log.info("=" * 60)

    t_start = datetime.now()

    # ── Load best params from step05 ──
    results_csv = RES_DIR / "step05_grid_results.csv"
    df_res = pd.read_csv(results_csv)
    rf_best = df_res[df_res["model"] == "rf"].dropna(subset=["pearson_r"]).nlargest(1, "pearson_r").iloc[0]
    cutoff   = float(rf_best["cutoff"])
    rna_d    = int(rf_best["rna_depth"])
    lig_d    = int(rf_best["lig_depth"])
    fp_size  = int(rf_best["fp_size"])
    feat     = int(rf_best["feat_set"])
    log.info(f"Best params: cutoff={cutoff}  rna_d={rna_d}  lig_d={lig_d}  fp={fp_size}  feat={feat}  grid_r={rf_best['pearson_r']:.4f}")

    c_str    = f"{cutoff:.1f}"
    npz_name = f"rlec_c{c_str}_r{rna_d}_l{lig_d}_s{fp_size}_f{feat}.npz"
    npz_path = FEAT_DIR / npz_name

    data = np.load(npz_path)
    X    = data["X"]
    y    = data["y"]
    ids  = data["ids"]

    # ── Subtype map ──
    df_meta  = pd.read_csv(DATA_CSV)[["pdb", "subtype"]]
    subtype_map = dict(zip(df_meta["pdb"], df_meta["subtype"]))
    subtypes = np.array([subtype_map.get(p, "unknown") for p in ids])
    unique_subtypes = sorted(set(subtypes))
    log.info(f"Subtypes: {dict(zip(*np.unique(subtypes, return_counts=True)))}")

    # Load Optuna LGB params
    with open(RES_DIR / "step07d_optuna_results.json") as f:
        lgb_params = json.load(f)["results"]["lgb_optuna"]["best_params"]
    log.info(f"Using LightGBM Optuna params: {lgb_params}")

    def fit_predict(X_tr, y_tr, X_te):
        m = lgb.LGBMRegressor(**lgb_params, random_state=42, verbose=-1, n_jobs=4)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)

    # ── Analysis 1: Standard 80/20 split — per-subtype breakdown ──
    log.info("\n--- ANALYSIS 1: Per-subtype on standard 80/20 split (LGB Optuna) ---")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(np.zeros(len(ids)), subtypes))

    y_pred_all = fit_predict(X[train_idx], y[train_idx], X[test_idx])
    y_test_all = y[test_idx]
    sub_test   = subtypes[test_idx]

    per_subtype_80_20 = {}
    for st in unique_subtypes:
        mask = sub_test == st
        if mask.sum() < 2:
            log.info(f"  {st:<20}: n={mask.sum()} (too few to compute r)")
            per_subtype_80_20[st] = {"n": int(mask.sum()), "pearson_r": None, "rmse": None}
            continue
        r, rmse = pearson_metrics(y_test_all[mask], y_pred_all[mask])
        log.info(f"  {st:<20}: n={mask.sum():2d}  r={r}  rmse={rmse}")
        per_subtype_80_20[st] = {"n": int(mask.sum()), "pearson_r": r, "rmse": rmse}

    r_all, rmse_all = pearson_metrics(y_test_all, y_pred_all)
    log.info(f"  {'OVERALL':<20}: n={len(y_test_all):2d}  r={r_all}  rmse={rmse_all}")

    # ── Analysis 2: Leave-One-Subtype-Out (LOSO) ──
    log.info("\n--- ANALYSIS 2: Leave-One-Subtype-Out (LOSO, LGB Optuna) ---")
    loso_results = {}
    for st in unique_subtypes:
        test_mask  = subtypes == st
        train_mask = ~test_mask
        n_test  = test_mask.sum()
        n_train = train_mask.sum()

        if n_test < 2:
            log.info(f"  {st:<20}: n_test={n_test} — skipped (too few)")
            loso_results[st] = {"n_train": int(n_train), "n_test": int(n_test),
                                "pearson_r": None, "rmse": None}
            continue

        y_pred = fit_predict(X[train_mask], y[train_mask], X[test_mask])
        r, rmse = pearson_metrics(y[test_mask], y_pred)
        log.info(f"  {st:<20}: n_train={n_train:3d}  n_test={n_test:2d}  r={r}  rmse={rmse}")
        loso_results[st] = {"n_train": int(n_train), "n_test": int(n_test),
                            "pearson_r": r, "rmse": rmse}

    # ── Save ──
    elapsed = (datetime.now() - t_start).total_seconds()
    out = {
        "timestamp"     : t_start.isoformat(),
        "elapsed_sec"   : elapsed,
        "model"         : "LightGBM (Optuna-tuned)",
        "eval_analysis1": "80/20 stratified split — per-subtype breakdown",
        "eval_analysis2": "Leave-One-Subtype-Out (LOSO)",
        "best_params"   : {"cutoff": cutoff, "rna_depth": rna_d, "lig_depth": lig_d,
                           "fp_size": fp_size, "feat_set": feat},
        "overall_80_20" : {"pearson_r": r_all, "rmse": rmse_all, "n_test": len(y_test_all)},
        "per_subtype_80_20": per_subtype_80_20,
        "loso"          : loso_results,
    }

    def to_python(obj):
        if isinstance(obj, dict):            return {str(k): to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):            return [to_python(v) for v in obj]
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        return obj

    out_path = RES_DIR / "step09_subtype_results.json"
    with open(out_path, "w") as f:
        json.dump(to_python(out), f, indent=2)
    with open(LOG_DIR / "step09_subtype.json", "w") as f:
        json.dump(to_python(out), f, indent=2)

    log.info(f"\nDone in {elapsed:.1f}s  |  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
