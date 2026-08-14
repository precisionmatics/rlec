"""
STEP 07d — LightGBM Hyperparameter Tuning via Optuna
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Uses Optuna TPE sampler to find best LightGBM hyperparameters.
Objective: maximize 5-fold OOF pooled Pearson r (faster than LOOCV for tuning).
Final evaluation of best params: LOOCV pooled r (n=143, honest metric).

Also tries distance-binned RLEC: instead of binary contacts, bins distances
into [0-3.5], [3.5-4.5], [4.5-5.5], [5.5-6.0] Å and uses binned count FP.

Outputs:
  results/step07d_optuna_results.json
  logs/step07d_optuna.json
"""

import sys, os, json, logging, warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

ROOT        = Path(__file__).parent.parent
FEAT_DIR    = ROOT / "data" / "features"
CONTACT_DIR = ROOT / "data" / "contacts"
RES_DIR     = ROOT / "results"
LOG_DIR     = ROOT / "logs"
DATA_CSV    = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")

N_TRIALS  = 150
N_FOLDS   = 5     # for Optuna objective (fast)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def oof5_r(X, y, subtypes, params):
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, subtypes):
        m = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=4)
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    r, _ = stats.pearsonr(y, y_pred)
    return float(r)


def loocv_r(X, y, params):
    loo    = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=4)
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    r, _ = stats.pearsonr(y, y_pred)
    rmse  = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return round(float(r), 4), round(rmse, 4)


def main():
    log.info("=" * 60)
    log.info("RLEC Step 07d — LightGBM Optuna Tuning + Distance-Binned FP")
    log.info("=" * 60)

    t_start = datetime.now()

    # ── Load best RLEC FP ──
    df_res   = pd.read_csv(RES_DIR / "step05_grid_results.csv")
    rf_best  = df_res[df_res["model"] == "rf"].nlargest(1, "pearson_r").iloc[0]
    npz_name = (f"rlec_c{rf_best['cutoff']:.1f}_r{int(rf_best['rna_depth'])}"
                f"_l{int(rf_best['lig_depth'])}_s{int(rf_best['fp_size'])}"
                f"_f{int(rf_best['feat_set'])}.npz")
    data     = np.load(FEAT_DIR / npz_name)
    X, y, ids = data["X"], data["y"], data["ids"]

    df_meta     = pd.read_csv(DATA_CSV)[["pdb", "subtype"]]
    subtype_map = dict(zip(df_meta["pdb"], df_meta["subtype"]))
    subtypes    = np.array([subtype_map.get(p, "unknown") for p in ids])
    log.info(f"FP: {npz_name}  X={X.shape}")

    results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1: Optuna tuning of LightGBM on best RLEC FP
    # ──────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- PART 1: Optuna ({N_TRIALS} trials, 5-fold OOF objective) ---")

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 3, 8),
            "num_leaves"       : trial.suggest_int("num_leaves", 10, 80),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.2, 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, 20),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return oof5_r(X, y, subtypes, params)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_trial_r = study.best_value
    log.info(f"  Best trial 5-fold r = {best_trial_r:.4f}")
    log.info(f"  Best params: {best_params}")

    # Evaluate best params on LOOCV
    log.info(f"  Running LOOCV with best params...")
    r_loocv, rmse_loocv = loocv_r(X, y, best_params)
    log.info(f"  LOOCV r = {r_loocv:.4f}  RMSE = {rmse_loocv:.4f}")

    results["lgb_optuna"] = {
        "best_trial_5fold_r": round(best_trial_r, 4),
        "loocv_r"           : r_loocv,
        "loocv_rmse"        : rmse_loocv,
        "best_params"       : best_params,
        "n_trials"          : N_TRIALS,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2: Distance-binned RLEC FP
    # ──────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- PART 2: Distance-Binned RLEC FP ---")
    log.info("  Builds separate count FP for each distance bin:")
    log.info("  bin0=[0-3.5Å] bin1=[3.5-4.5Å] bin2=[4.5-5.5Å] bin3=[5.5-6.0Å]")
    log.info("  Then concatenates → 4×4096 = 16384-dim FP")

    # Load contacts
    pkl_files = sorted(CONTACT_DIR.glob("*_contacts.pkl"))
    complexes = []
    for f in pkl_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        complexes.append(d)

    # Build distance-binned FP using RLEC hashes from the best npz
    # We re-use the existing RLEC bit computation logic
    # Strategy: for each contact, assign to bin, fold hash into that bin's subvector
    BINS     = [(0.0, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 6.0)]
    FP_SIZE  = 4096
    N_BINS   = len(BINS)
    pdb_order = list(ids)

    # Build pdb → complex index
    pdb_to_cdata = {c["pdb_id"]: c for c in complexes}

    # Load RLEC hash info — recompute bits from contacts + hashes
    # We need the contact-level hashes. Rather than recomputing from scratch,
    # use the existing npz (count FP at cutoff=6.0) as base features,
    # and build a simpler distance-binned version using element pairs.
    X_binned = np.zeros((len(pdb_order), FP_SIZE * N_BINS), dtype=np.float32)

    for pi, pdb_id in enumerate(pdb_order):
        cdata = pdb_to_cdata.get(pdb_id)
        if cdata is None:
            continue
        rna_atoms = {a["idx"]: a for a in cdata["rna_atoms"]}
        lig_atoms = {a["idx"]: a for a in cdata["lig_atoms"]}
        contacts_60 = cdata["contacts"].get(6.0, [])

        for rna_idx, lig_idx, dist in contacts_60:
            ra = rna_atoms.get(rna_idx)
            la = lig_atoms.get(lig_idx)
            if ra is None or la is None:
                continue

            # Find distance bin
            bin_idx = N_BINS - 1
            for bi, (lo, hi) in enumerate(BINS):
                if dist <= hi:
                    bin_idx = bi
                    break

            # Hash: RNA atom features + ligand atom features + distance bin
            h = hash((ra["atomic_num"], ra.get("nuc_type", 0), ra.get("pbs", 0),
                      ra["n_heavy_nbrs"], ra["is_aromatic"],
                      la["atomic_num"], la["n_heavy_nbrs"],
                      la["is_aromatic"], la["in_ring"],
                      bin_idx)) & 0xFFFFFFFF

            offset = bin_idx * FP_SIZE
            X_binned[pi][offset + (h % FP_SIZE)] += 1

    log.info(f"  X_binned shape: {X_binned.shape}  "
             f"nonzero rows: {(X_binned.sum(1) > 0).sum()}")

    # Evaluate with Optuna-best LGB params
    log.info(f"  Evaluating distance-binned FP (LOOCV, LGB optuna params)...")
    r_bin, rmse_bin = loocv_r(X_binned, y, best_params)
    log.info(f"  Distance-binned LOOCV r = {r_bin:.4f}  RMSE = {rmse_bin:.4f}")

    # Also try concatenation: original RLEC + distance-binned
    X_concat = np.hstack([X, X_binned])
    log.info(f"  Evaluating concat (RLEC + distance-binned) shape={X_concat.shape}...")
    r_cat, rmse_cat = loocv_r(X_concat, y, best_params)
    log.info(f"  Concat LOOCV r = {r_cat:.4f}  RMSE = {rmse_cat:.4f}")

    results["distance_binned"] = {
        "fp_shape"    : list(X_binned.shape),
        "loocv_r"     : r_bin,
        "loocv_rmse"  : rmse_bin,
    }
    results["rlec_plus_binned"] = {
        "fp_shape"    : list(X_concat.shape),
        "loocv_r"     : r_cat,
        "loocv_rmse"  : rmse_cat,
    }

    # ── Final summary ──
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  LightGBM default (step07c):          r=0.6938")
    log.info(f"  LightGBM Optuna ({N_TRIALS} trials, LOOCV): r={r_loocv:.4f}")
    log.info(f"  Distance-binned RLEC (LOOCV):         r={r_bin:.4f}")
    log.info(f"  RLEC + Distance-binned (LOOCV):       r={r_cat:.4f}")
    best_overall = max(0.6938, r_loocv, r_bin, r_cat)
    log.info(f"\n  BEST OVERALL: r={best_overall:.4f}")

    # ── Save ──
    elapsed = (datetime.now() - t_start).total_seconds()
    out = {"timestamp": t_start.isoformat(), "elapsed_sec": elapsed,
           "baseline_lgb_default": 0.6938, "results": results}

    def to_python(obj):
        if isinstance(obj, dict):   return {str(k): to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [to_python(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    out_path = RES_DIR / "step07d_optuna_results.json"
    with open(out_path, "w") as f:
        json.dump(to_python(out), f, indent=2)
    with open(LOG_DIR / "step07d_optuna.json", "w") as f:
        json.dump(to_python(out), f, indent=2)

    log.info(f"\nDone in {elapsed:.1f}s  |  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
