"""
STEP 07 — 10-Fold Stratified Cross-Validation
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Selects best parameters from step05 grid results, then runs 10-fold
stratified CV (by RNA subtype) on those parameters for RF, NN, and linear.

This is the honest performance estimate that goes in the paper —
not the grid-search best which is inflated from searching 4032 combos.

Outputs:
  results/step07_cv_results.json   — per-fold metrics + mean ± SD
  logs/step07_cv.json              — run log
"""

import sys, os, json, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

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


def pearson(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan, np.nan, np.nan
    r, _ = stats.pearsonr(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sd   = float(np.std(y_true - y_pred))
    return round(float(r), 4), round(rmse, 4), round(sd, 4)


def run_cv(X, y, subtypes, model_name, n_splits=10):
    """Run stratified k-fold CV, return per-fold and summary metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, subtypes)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if model_name == "rf":
            m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)

        elif model_name == "nn":
            scaler = StandardScaler(with_mean=False)
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            m = MLPRegressor(hidden_layer_sizes=(100, 100), activation="relu",
                             solver="adam", max_iter=500,
                             early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=15, random_state=42)
            m.fit(X_tr_s, y_tr)
            y_pred = m.predict(X_te_s)

        elif model_name == "linear":
            m = SGDRegressor(loss="huber", penalty="elasticnet", l1_ratio=0.5,
                             epsilon=0.1, max_iter=2000, tol=1e-4, random_state=42)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)

        r, rmse, sd = pearson(y_te, y_pred)
        fold_results.append({"fold": fold+1, "n_test": len(test_idx),
                              "pearson_r": r, "rmse": rmse, "sd": sd})
        log.info(f"    Fold {fold+1:2d}: r={r:.4f}  rmse={rmse:.4f}  n_test={len(test_idx)}")

    rs   = [f["pearson_r"] for f in fold_results]
    rmses = [f["rmse"] for f in fold_results]
    summary = {
        "pearson_r_mean": round(float(np.mean(rs)), 4),
        "pearson_r_std" : round(float(np.std(rs)), 4),
        "pearson_r_min" : round(float(np.min(rs)), 4),
        "pearson_r_max" : round(float(np.max(rs)), 4),
        "rmse_mean"     : round(float(np.mean(rmses)), 4),
        "rmse_std"      : round(float(np.std(rmses)), 4),
    }
    return fold_results, summary


def main():
    log.info("=" * 60)
    log.info("RLEC Step 07 — 10-Fold Stratified Cross-Validation")
    log.info("=" * 60)

    t_start = datetime.now()

    # ── Load step05 results ──
    results_csv = RES_DIR / "step05_grid_results.csv"
    if not results_csv.exists():
        log.error("step05_grid_results.csv not found — run step05 first")
        return 1

    df = pd.read_csv(results_csv)
    log.info(f"Loaded {len(df)} rows from step05 results")

    # ── Find best params per model ──
    best_params = {}
    for mdl in ["rf", "nn", "linear"]:
        sub = df[df["model"] == mdl].dropna(subset=["pearson_r"])
        b   = sub.nlargest(1, "pearson_r").iloc[0]
        best_params[mdl] = {
            "cutoff"   : float(b["cutoff"]),
            "rna_depth": int(b["rna_depth"]),
            "lig_depth": int(b["lig_depth"]),
            "fp_size"  : int(b["fp_size"]),
            "feat_set" : int(b["feat_set"]),
            "grid_r"   : float(b["pearson_r"]),
        }
        log.info(f"  Best {mdl}: r={b['pearson_r']:.4f}  "
                 f"cutoff={b['cutoff']}  rna_d={int(b['rna_depth'])}  "
                 f"lig_d={int(b['lig_depth'])}  fp={int(b['fp_size'])}  feat={int(b['feat_set'])}")

    # ── Load subtype map ──
    df_meta = pd.read_csv(DATA_CSV)[["pdb", "subtype"]]
    subtype_map = dict(zip(df_meta["pdb"], df_meta["subtype"]))

    # ── Run CV for each model ──
    all_results = {}

    for mdl in ["rf", "nn", "linear"]:
        p = best_params[mdl]
        c_str = f"{p['cutoff']:.1f}"
        npz_name = (f"rlec_c{c_str}_r{p['rna_depth']}_l{p['lig_depth']}"
                    f"_s{p['fp_size']}_f{p['feat_set']}.npz")
        npz_path = FEAT_DIR / npz_name

        if not npz_path.exists():
            log.error(f"NPZ not found: {npz_path}")
            continue

        log.info(f"\nLoading {npz_name} ...")
        data = np.load(npz_path)
        X    = data["X"]
        y    = data["y"]
        ids  = data["ids"]

        subtypes = np.array([subtype_map.get(pid, "unknown") for pid in ids])
        log.info(f"  X={X.shape}  y={y.shape}  subtypes: {dict(zip(*np.unique(subtypes, return_counts=True)))}")

        log.info(f"\n--- {mdl.upper()} 10-fold CV (grid best r={p['grid_r']:.4f}) ---")
        fold_results, summary = run_cv(X, y, subtypes, mdl)

        log.info(f"  → mean r = {summary['pearson_r_mean']:.4f} ± {summary['pearson_r_std']:.4f}"
                 f"  (range [{summary['pearson_r_min']:.4f}, {summary['pearson_r_max']:.4f}])")
        log.info(f"  → mean RMSE = {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")

        all_results[mdl] = {
            "best_params" : p,
            "npz_file"    : npz_name,
            "folds"       : fold_results,
            "summary"     : summary,
        }

    # ── Also run CV on the single globally best combo ──
    log.info("\n--- GLOBAL BEST COMBO CV ---")
    best_row = df.dropna(subset=["pearson_r"]).nlargest(1, "pearson_r").iloc[0]
    log.info(f"  Global best: model={best_row['model']}  r={best_row['pearson_r']:.4f}  "
             f"cutoff={best_row['cutoff']}  rna_d={int(best_row['rna_depth'])}  "
             f"lig_d={int(best_row['lig_depth'])}  fp={int(best_row['fp_size'])}  feat={int(best_row['feat_set'])}")

    c_str    = f"{best_row['cutoff']:.1f}"
    npz_name = (f"rlec_c{c_str}_r{int(best_row['rna_depth'])}_l{int(best_row['lig_depth'])}"
                f"_s{int(best_row['fp_size'])}_f{int(best_row['feat_set'])}.npz")
    npz_path = FEAT_DIR / npz_name
    if npz_path.exists():
        data     = np.load(npz_path)
        X, y, ids = data["X"], data["y"], data["ids"]
        subtypes = np.array([subtype_map.get(pid, "unknown") for pid in ids])
        fold_results, summary = run_cv(X, y, subtypes, str(best_row["model"]))
        log.info(f"  → mean r = {summary['pearson_r_mean']:.4f} ± {summary['pearson_r_std']:.4f}")
        all_results["global_best"] = {
            "best_params": {k: best_row[k] for k in
                            ["model","cutoff","rna_depth","lig_depth","fp_size","feat_set","pearson_r"]},
            "npz_file"   : npz_name,
            "folds"      : fold_results,
            "summary"    : summary,
        }

    # ── Save ──
    elapsed = (datetime.now() - t_start).total_seconds()
    out = {"timestamp": t_start.isoformat(), "elapsed_sec": elapsed,
           "n_folds": 10, "results": all_results}

    def to_python(obj):
        if isinstance(obj, dict):
            return {str(k): to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    out_path = RES_DIR / "step07_cv_results.json"
    with open(out_path, "w") as f:
        json.dump(to_python(out), f, indent=2)
    with open(LOG_DIR / "step07_cv.json", "w") as f:
        json.dump(to_python(out), f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"Done in {elapsed:.1f}s")
    log.info(f"Results: {out_path}")

    log.info(f"\n{'='*60}")
    log.info("FINAL SUMMARY (10-fold CV — honest numbers for paper)")
    log.info(f"{'='*60}")
    for mdl, res in all_results.items():
        s = res["summary"]
        log.info(f"  {mdl:12s}: r = {s['pearson_r_mean']:.4f} ± {s['pearson_r_std']:.4f}  "
                 f"RMSE = {s['rmse_mean']:.4f} ± {s['rmse_std']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
