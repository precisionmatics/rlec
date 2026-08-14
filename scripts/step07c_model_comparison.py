"""
STEP 07c — Full Model Comparison (LOOCV)
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Tries all strong regression models on the best RLEC FP (cutoff=6.0, rna_d=6,
lig_d=3, fp=4096, feat=1) using LOOCV pooled Pearson r (n=143).

Models:
  - RandomForest (best config from 07b: max_features=sqrt, n=500)
  - ExtraTrees
  - XGBoost
  - LightGBM
  - SVR (RBF kernel)
  - SVR (linear kernel)
  - GradientBoosting
  - Stacked ensemble (RF + XGB + LGB → Ridge meta)

Outputs:
  results/step07c_model_comparison.json
  logs/step07c_model_comparison.log
"""

import sys, os, json, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import xgboost as xgb
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


def loocv_pooled_r(X, y, model_fn, scale=False):
    """LOOCV: pool all OOF predictions → compute Pearson r on full n=143."""
    loo    = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)
    scaler = StandardScaler(with_mean=False) if scale else None

    for tr, te in loo.split(X):
        X_tr, X_te = X[tr], X[te]
        if scaler is not None:
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
        m = model_fn()
        m.fit(X_tr, y[tr])
        y_pred[te] = m.predict(X_te)

    r, _ = stats.pearsonr(y, y_pred)
    rmse  = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return round(float(r), 4), round(rmse, 4), y_pred


def main():
    log.info("=" * 60)
    log.info("RLEC Step 07c — Full Model Comparison (LOOCV, n=143)")
    log.info("=" * 60)

    t_start = datetime.now()

    # ── Load best RLEC FP ──
    df_res   = pd.read_csv(RES_DIR / "step05_grid_results.csv")
    rf_best  = df_res[df_res["model"] == "rf"].nlargest(1, "pearson_r").iloc[0]
    npz_name = (f"rlec_c{rf_best['cutoff']:.1f}_r{int(rf_best['rna_depth'])}"
                f"_l{int(rf_best['lig_depth'])}_s{int(rf_best['fp_size'])}"
                f"_f{int(rf_best['feat_set'])}.npz")
    npz_path = FEAT_DIR / npz_name
    data     = np.load(npz_path)
    X, y, ids = data["X"], data["y"], data["ids"]
    log.info(f"FP: {npz_name}  X={X.shape}")

    results  = {}
    best_r   = 0
    best_mdl = ""

    # ── Define all models ──
    models = [
        ("RF (max_feat=sqrt, n=500)",
         lambda: RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                       n_jobs=-1, random_state=42), False),
        ("ExtraTrees (n=500)",
         lambda: ExtraTreesRegressor(n_estimators=500, max_features="sqrt",
                                     n_jobs=-1, random_state=42), False),
        ("XGBoost",
         lambda: xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                   max_depth=6, subsample=0.8,
                                   colsample_bytree=0.5, min_child_weight=3,
                                   reg_alpha=0.1, reg_lambda=1.0,
                                   n_jobs=-1, random_state=42,
                                   verbosity=0), False),
        # XGBoost dart skipped — too slow with LOOCV
        ("LightGBM",
         lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                    max_depth=6, num_leaves=31,
                                    subsample=0.8, colsample_bytree=0.5,
                                    min_child_samples=5, reg_alpha=0.1,
                                    n_jobs=-1, random_state=42,
                                    verbose=-1), False),
        ("LightGBM (tuned)",
         lambda: lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.02,
                                    max_depth=5, num_leaves=20,
                                    subsample=0.7, colsample_bytree=0.4,
                                    min_child_samples=3, reg_alpha=0.5,
                                    reg_lambda=0.5, n_jobs=-1,
                                    random_state=42, verbose=-1), False),
        ("SVR (RBF)",
         lambda: SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1), True),
        ("SVR (linear)",
         lambda: SVR(kernel="linear", C=1.0, epsilon=0.1), True),
        ("SVR (RBF, C=100)",
         lambda: SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.05), True),
        ("GradientBoosting",
         lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                            max_depth=4, subsample=0.8,
                                            random_state=42), False),
    ]

    all_preds = {}

    for name, model_fn, scale in models:
        log.info(f"\n  [{name}]")
        r, rmse, y_pred = loocv_pooled_r(X, y, model_fn, scale=scale)
        marker = " ◄ BEST" if r > best_r else ""
        log.info(f"    LOOCV r={r:.4f}  RMSE={rmse:.4f}{marker}")
        if r > best_r:
            best_r   = r
            best_mdl = name
        results[name] = {"loocv_r": r, "loocv_rmse": rmse}
        all_preds[name] = y_pred

    # ── Stacked Ensemble: RF + XGB + LGB → Ridge meta ──
    log.info("\n  [Stacked Ensemble: RF + XGB + LGB → Ridge]")
    base_names = ["RF (max_feat=sqrt, n=500)", "XGBoost", "LightGBM"]
    loo = LeaveOneOut()
    meta_X   = np.zeros((len(y), len(base_names)))
    meta_pred = np.zeros(len(y))

    for fi, bname in enumerate(base_names):
        meta_X[:, fi] = all_preds[bname]

    # Ridge meta-learner via LOOCV on OOF predictions
    for tr, te in loo.split(meta_X):
        meta = Ridge(alpha=1.0)
        meta.fit(meta_X[tr], y[tr])
        meta_pred[te] = meta.predict(meta_X[te])

    r_stack, _ = stats.pearsonr(y, meta_pred)
    rmse_stack  = float(np.sqrt(np.mean((y - meta_pred) ** 2)))
    r_stack     = round(float(r_stack), 4)
    rmse_stack  = round(rmse_stack, 4)
    marker = " ◄ BEST" if r_stack > best_r else ""
    log.info(f"    LOOCV r={r_stack:.4f}  RMSE={rmse_stack:.4f}{marker}")
    if r_stack > best_r:
        best_r   = r_stack
        best_mdl = "Stacked (RF+XGB+LGB)"
    results["Stacked (RF+XGB+LGB→Ridge)"] = {"loocv_r": r_stack, "loocv_rmse": rmse_stack}

    # ── Simple average ensemble ──
    log.info("\n  [Average Ensemble: RF + XGB + LGB]")
    avg_pred = np.mean([all_preds[n] for n in base_names], axis=0)
    r_avg, _ = stats.pearsonr(y, avg_pred)
    rmse_avg  = float(np.sqrt(np.mean((y - avg_pred) ** 2)))
    r_avg, rmse_avg = round(float(r_avg), 4), round(rmse_avg, 4)
    marker = " ◄ BEST" if r_avg > best_r else ""
    log.info(f"    LOOCV r={r_avg:.4f}  RMSE={rmse_avg:.4f}{marker}")
    if r_avg > best_r:
        best_r   = r_avg
        best_mdl = "Average (RF+XGB+LGB)"
    results["Average (RF+XGB+LGB)"] = {"loocv_r": r_avg, "loocv_rmse": rmse_avg}

    # ── Final summary ──
    log.info(f"\n{'='*60}")
    log.info("FULL MODEL COMPARISON SUMMARY (LOOCV pooled r, n=143)")
    log.info(f"{'='*60}")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["loocv_r"], reverse=True)
    for name, res in sorted_results:
        bar = "█" * int(res["loocv_r"] * 20)
        log.info(f"  {name:<40}: r={res['loocv_r']:.4f}  {bar}")
    log.info(f"\n  BEST MODEL: {best_mdl}  r={best_r:.4f}")

    # ── Save ──
    elapsed  = (datetime.now() - t_start).total_seconds()
    out      = {"timestamp": t_start.isoformat(), "elapsed_sec": elapsed,
                "fp": npz_name, "n": int(len(y)),
                "best_model": best_mdl, "best_loocv_r": best_r,
                "results": results}

    def to_python(obj):
        if isinstance(obj, dict):   return {str(k): to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [to_python(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    out_path = RES_DIR / "step07c_model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(to_python(out), f, indent=2)
    with open(LOG_DIR / "step07c_model_comparison.json", "w") as f:
        json.dump(to_python(out), f, indent=2)

    log.info(f"\nDone in {elapsed:.1f}s  |  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
