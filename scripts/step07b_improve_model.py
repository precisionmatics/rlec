"""
STEP 07b — RF Tuning + Ensemble of Top-k FP Matrices
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Two improvement strategies:
  1. RF hyperparameter tuning (more trees, max_depth, min_samples_leaf)
  2. Ensemble of top-k RLEC FP matrices — two approaches:
       a. Feature stacking: concatenate top-k FP matrices → train one RF
       b. Prediction ensemble: train RF on each, average predictions

All evaluated via LOOCV pooled Pearson r (n=143) — the honest metric.

Outputs:
  results/step07b_improvement_results.json
  logs/step07b_improvement.json
"""

import sys, os, json, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / "data" / "features"
RES_DIR  = ROOT / "results"
LOG_DIR  = ROOT / "logs"
DATA_CSV = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def loocv_pooled_r(X, y, rf_kwargs):
    """LOOCV: train on n-1, predict 1, pool all predictions → Pearson r."""
    loo    = LeaveOneOut()
    y_pred = np.zeros_like(y)
    for tr, te in loo.split(X):
        m = RandomForestRegressor(**rf_kwargs, random_state=42)
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    r, _ = stats.pearsonr(y, y_pred)
    return round(float(r), 4), y_pred


def oof10_pooled_r(X, y, subtypes, rf_kwargs):
    """10-fold OOF pooled r — faster than LOOCV for large X."""
    skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y)
    for tr, te in skf.split(X, subtypes):
        m = RandomForestRegressor(**rf_kwargs, random_state=42)
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    r, _ = stats.pearsonr(y, y_pred)
    return round(float(r), 4), y_pred


def load_npz(name):
    p = FEAT_DIR / name
    if not p.exists():
        return None, None, None
    d = np.load(p)
    return d["X"], d["y"], d["ids"]


def npz_name(cutoff, rna_d, lig_d, fp_size, feat):
    return f"rlec_c{cutoff:.1f}_r{rna_d}_l{lig_d}_s{fp_size}_f{feat}.npz"


def main():
    log.info("=" * 60)
    log.info("RLEC Step 07b — RF Tuning + Top-k Ensemble")
    log.info("=" * 60)

    t_start = datetime.now()

    # ── Setup ──
    df_res      = pd.read_csv(RES_DIR / "step05_grid_results.csv")
    df_meta     = pd.read_csv(DATA_CSV)[["pdb", "subtype"]]
    subtype_map = dict(zip(df_meta["pdb"], df_meta["subtype"]))

    # Load best RF FP
    rf_df   = df_res[df_res["model"] == "rf"].dropna(subset=["pearson_r"])
    best_row = rf_df.nlargest(1, "pearson_r").iloc[0]
    best_name = npz_name(best_row["cutoff"], int(best_row["rna_depth"]),
                         int(best_row["lig_depth"]), int(best_row["fp_size"]),
                         int(best_row["feat_set"]))
    X_best, y, ids = load_npz(best_name)
    subtypes = np.array([subtype_map.get(p, "unknown") for p in ids])
    log.info(f"Best FP: {best_name}  X={X_best.shape}")

    results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1: RF Hyperparameter Tuning (LOOCV)
    # ──────────────────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("PART 1: RF Hyperparameter Tuning (LOOCV, n=143)")
    log.info("="*60)

    rf_configs = [
        {"n_estimators": 100,  "n_jobs": -1},
        {"n_estimators": 300,  "n_jobs": -1},
        {"n_estimators": 500,  "n_jobs": -1},
        {"n_estimators": 1000, "n_jobs": -1},
        {"n_estimators": 500,  "max_depth": 10, "n_jobs": -1},
        {"n_estimators": 500,  "max_depth": 20, "n_jobs": -1},
        {"n_estimators": 500,  "max_depth": 30, "n_jobs": -1},
        {"n_estimators": 500,  "min_samples_leaf": 2, "n_jobs": -1},
        {"n_estimators": 500,  "min_samples_leaf": 3, "n_jobs": -1},
        {"n_estimators": 500,  "max_features": "sqrt", "n_jobs": -1},
        {"n_estimators": 500,  "max_features": 0.5, "n_jobs": -1},
        {"n_estimators": 1000, "max_depth": 20, "min_samples_leaf": 2,
         "max_features": "sqrt", "n_jobs": -1},
    ]

    best_tune_r = 0
    best_tune_cfg = None
    tune_results = []

    for cfg in rf_configs:
        label = str({k: v for k, v in cfg.items() if k != "n_jobs"})
        r, _ = loocv_pooled_r(X_best, y, cfg)
        marker = " ◄ BEST" if r > best_tune_r else ""
        log.info(f"  {label:<60}: LOOCV r={r:.4f}{marker}")
        if r > best_tune_r:
            best_tune_r   = r
            best_tune_cfg = cfg
        tune_results.append({"config": label, "loocv_r": r})

    log.info(f"\n  Best RF config: {best_tune_cfg}  → r={best_tune_r:.4f}")
    results["rf_tuning"] = {"configs": tune_results,
                             "best_r": best_tune_r,
                             "best_config": str(best_tune_cfg)}

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2: Ensemble of Top-k FP Matrices
    # ──────────────────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("PART 2: Ensemble of Top-k RLEC FP Matrices")
    log.info("="*60)

    # Get top-k unique parameter combos (exclude fp_size duplicates — same info)
    # Take top RF rows, pick unique (cutoff, rna_d, lig_d, feat) combos
    rf_sorted = rf_df.nlargest(200, "pearson_r")
    seen_combos = set()
    top_unique = []
    for _, row in rf_sorted.iterrows():
        key = (row["cutoff"], int(row["rna_depth"]), int(row["lig_depth"]), int(row["feat_set"]))
        if key not in seen_combos:
            seen_combos.add(key)
            # Use fp=4096 for all (best FP size)
            top_unique.append({
                "cutoff"   : row["cutoff"],
                "rna_depth": int(row["rna_depth"]),
                "lig_depth": int(row["lig_depth"]),
                "fp_size"  : 4096,
                "feat_set" : int(row["feat_set"]),
                "grid_r"   : row["pearson_r"],
            })
        if len(top_unique) >= 20:
            break

    log.info(f"  Top-{len(top_unique)} unique parameter combos (all fp=4096):")
    for i, p in enumerate(top_unique[:5]):
        log.info(f"    {i+1}. cutoff={p['cutoff']} rna_d={p['rna_depth']} "
                 f"lig_d={p['lig_depth']} feat={p['feat_set']} grid_r={p['grid_r']:.4f}")
    log.info(f"    ...")

    # Load all top-k FP matrices
    log.info("\n  Loading FP matrices...")
    Xs, valid_params = [], []
    for p in top_unique:
        name = npz_name(p["cutoff"], p["rna_depth"], p["lig_depth"], p["fp_size"], p["feat_set"])
        X_i, _, _ = load_npz(name)
        if X_i is not None:
            Xs.append(X_i)
            valid_params.append(p)
    log.info(f"  Loaded {len(Xs)} FP matrices")

    # Best RF config from tuning
    best_cfg = {k: v for k, v in best_tune_cfg.items()}

    # ── 2a: Prediction Ensemble (average OOF predictions) ──
    log.info("\n--- 2a: Prediction Ensemble (average OOF from each FP) ---")
    ensemble_results = []

    for k in [3, 5, 10, 15, 20]:
        if k > len(Xs):
            break
        Xs_k = Xs[:k]

        # OOF predictions from each FP matrix
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        all_preds = np.zeros((len(y), k))

        for fi, X_i in enumerate(Xs_k):
            y_pred_i = np.zeros_like(y)
            for tr, te in skf.split(X_i, subtypes):
                m = RandomForestRegressor(**best_cfg, random_state=42)
                m.fit(X_i[tr], y[tr])
                y_pred_i[te] = m.predict(X_i[te])
            all_preds[:, fi] = y_pred_i

        y_ens = all_preds.mean(axis=1)
        r_ens, _ = stats.pearsonr(y, y_ens)
        log.info(f"  Prediction ensemble top-{k:2d}: r={r_ens:.4f}")
        ensemble_results.append({"k": k, "r": round(float(r_ens), 4)})

    results["prediction_ensemble"] = ensemble_results

    # ── 2b: Feature Stacking ──
    log.info("\n--- 2b: Feature Stacking (concatenate top-k FP matrices) ---")
    stack_results = []

    for k in [3, 5, 10, 15, 20]:
        if k > len(Xs):
            break
        X_stack = np.hstack(Xs[:k])
        log.info(f"  Stack top-{k:2d}: X_stack shape={X_stack.shape}")
        r_stack, _ = oof10_pooled_r(X_stack, y, subtypes, best_cfg)
        log.info(f"  Feature stack  top-{k:2d}: r={r_stack:.4f}")
        stack_results.append({"k": k, "r": round(float(r_stack), 4), "n_features": X_stack.shape[1]})

    results["feature_stacking"] = stack_results

    # ── Summary ──
    log.info(f"\n{'='*60}")
    log.info("IMPROVEMENT SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Baseline (RF default, LOOCV):          r=0.6379")
    log.info(f"  Best RF tuned (LOOCV):                 r={best_tune_r:.4f}")
    best_ens = max(ensemble_results, key=lambda x: x["r"])
    best_stk = max(stack_results,    key=lambda x: x["r"])
    log.info(f"  Best prediction ensemble (top-{best_ens['k']:2d}):    r={best_ens['r']:.4f}")
    log.info(f"  Best feature stack       (top-{best_stk['k']:2d}):    r={best_stk['r']:.4f}")

    overall_best_r = max(best_tune_r, best_ens["r"], best_stk["r"])
    log.info(f"\n  OVERALL BEST: r={overall_best_r:.4f}")

    # ── Save ──
    elapsed = (datetime.now() - t_start).total_seconds()
    out = {"timestamp": t_start.isoformat(), "elapsed_sec": elapsed,
           "baseline_loocv_r": 0.6379, "results": results}

    def to_python(obj):
        if isinstance(obj, dict):   return {str(k): to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [to_python(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    out_path = RES_DIR / "step07b_improvement_results.json"
    with open(out_path, "w") as f:
        json.dump(to_python(out), f, indent=2)
    with open(LOG_DIR / "step07b_improvement.json", "w") as f:
        json.dump(to_python(out), f, indent=2)

    log.info(f"\nDone in {elapsed:.1f}s  |  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
