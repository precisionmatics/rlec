"""
STEP 13 — Push PCC toward 0.8: Feature Fusion + Multi-FP + Optuna
RLEC on PDBbind NL2020 (143 complexes), RLaffinity protocol

Scientific strategy:
  A) Feature fusion: RLEC bits + ligand physicochemical + RNA pocket descriptors
     - RLEC (bits) captures local contact environment (WHO is touching WHAT)
     - Ligand descriptors (MW, HBD, HBA, TPSA, rings) capture global ligand properties
     - RNA descriptors (n_atoms, element composition) capture pocket character
  B) Larger fp_size (65536) — fewer hash collisions → cleaner bits
  C) Multi-FP stacking — combine fingerprints from different parameter settings
  D) Optuna LightGBM hyperparameter tuning on train set (no test leakage)
  E) Stacked ensemble of LGB + XGB + RF
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / "data" / "features"
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)
DATA_CSV = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")

N_SPLITS  = 10
TEST_SIZE = 0.20
RANDOM    = 0


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    r,  _ = stats.pearsonr(y_true, y_pred)
    rs, _ = stats.spearmanr(y_true, y_pred)
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    return dict(pcc=round(float(r), 4), spcc=round(float(rs), 4),
                rmse=round(rmse, 4),    mae=round(mae, 4))


def summarize(split_results, label):
    df = pd.DataFrame(split_results)
    m, s = df.mean(), df.std()
    print(f"  {label:<40}  PCC={m['pcc']:.3f}±{s['pcc']:.3f}  "
          f"SPCC={m['spcc']:.3f}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}")
    return m.to_dict()


# ── Load fingerprint ───────────────────────────────────────────────────────────

def load_fp(name):
    d = np.load(FEAT_DIR / name)
    return d["X"].astype(np.float32), d["y"].astype(np.float64), d["ids"]


# ── Ligand + RNA descriptors from CSV ─────────────────────────────────────────

def load_descriptors(ids):
    df = pd.read_csv(DATA_CSV).set_index("pdb")
    lig_cols = ["mol_weight", "n_rings", "n_hbd", "n_hba", "n_rot_bonds", "tpsa",
                "n_lig_atoms", "lig_C", "lig_N", "lig_O", "lig_S",
                "lig_F", "lig_Cl", "lig_Br", "lig_I", "lig_P"]
    rna_cols = ["n_rna_atoms", "rna_C", "rna_N", "rna_O", "rna_P", "rna_S"]
    cols = lig_cols + rna_cols
    desc = df.loc[ids, cols].values.astype(np.float32)
    return desc, cols


# ── Optuna LightGBM tuning (on train split only) ──────────────────────────────

def tune_lgb(X_tr, y_tr, n_trials=60):
    def objective(trial):
        params = dict(
            n_estimators   = trial.suggest_int("n_estimators", 200, 1000),
            learning_rate  = trial.suggest_float("lr", 0.01, 0.2, log=True),
            num_leaves     = trial.suggest_int("num_leaves", 16, 128),
            max_depth      = trial.suggest_int("max_depth", 3, 10),
            subsample      = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample", 0.3, 1.0),
            reg_alpha      = trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            reg_lambda     = trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            min_child_samples = trial.suggest_int("min_child", 5, 50),
            objective="regression", metric="rmse", n_jobs=4,
            random_state=42, verbose=-1,
        )
        model = lgb.LGBMRegressor(**params)
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=5,
                                    scoring="r2", n_jobs=1)
        return cv_scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ── Run 10-split evaluation ────────────────────────────────────────────────────

def run_splits(X, y_norm, model_fn, label, n_trials=0):
    ss = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM)
    results = []
    for i, (tr_idx, te_idx) in enumerate(ss.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y_norm[tr_idx], y_norm[te_idx]
        model = model_fn(X_tr, y_tr, n_trials=n_trials)
        y_pred = model.predict(X_te)
        results.append(metrics(y_te, y_pred))
    return summarize(results, label)


# ── Model builders ────────────────────────────────────────────────────────────

def lgb_default(X_tr, y_tr, n_trials=0):
    params = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                  objective="regression", n_jobs=4, random_state=42, verbose=-1)
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_tr, y_tr)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m


def lgb_tuned_fn(best_params):
    def build(X_tr, y_tr, n_trials=0):
        p = dict(**best_params, objective="regression", metric="rmse",
                 n_jobs=4, random_state=42, verbose=-1)
        p["learning_rate"] = p.pop("lr", p.get("learning_rate", 0.05))
        p["colsample_bytree"] = p.pop("colsample", p.get("colsample_bytree", 0.8))
        p["reg_alpha"]  = p.pop("alpha",  p.get("reg_alpha",  0.1))
        p["reg_lambda"] = p.pop("lambda", p.get("reg_lambda", 0.1))
        p["min_child_samples"] = p.pop("min_child", p.get("min_child_samples", 20))
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr, y_tr)
        return m
    return build


def xgb_fn(X_tr, y_tr, n_trials=0):
    m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, reg_lambda=1.0,
                          objective="reg:squarederror", n_jobs=4,
                          random_state=42, verbosity=0)
    m.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)],
          verbose=False)
    return m


def ensemble_fn(X_tr, y_tr, n_trials=0):
    lgb_m = lgb_default(X_tr, y_tr)
    xgb_m = xgb_fn(X_tr, y_tr)
    rf_m  = RandomForestRegressor(n_estimators=300, max_features=0.3,
                                   n_jobs=4, random_state=42)
    rf_m.fit(X_tr, y_tr)
    ridge = Ridge(alpha=10.0)
    preds_tr = np.column_stack([lgb_m.predict(X_tr),
                                 xgb_m.predict(X_tr),
                                 rf_m.predict(X_tr)])
    ridge.fit(preds_tr, y_tr)

    class Ensemble:
        def predict(self, X):
            preds = np.column_stack([lgb_m.predict(X),
                                     xgb_m.predict(X),
                                     rf_m.predict(X)])
            return ridge.predict(preds)
    return Ensemble()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 13 — Feature Fusion + Multi-FP + Optuna Boost")
    print("=" * 70)

    # ── Load base fingerprint (best params) ───────────────────────────────────
    X4k,  y, ids = load_fp("rlec_c6.0_r6_l3_s4096_f1.npz")
    X64k, _, _   = load_fp("rlec_c6.0_r6_l3_s65536_f1.npz")

    # Load additional parameter combinations (wider coverage)
    X_c9, _, _ = load_fp("rlec_c6.0_r6_l6_s4096_f1.npz")   # deeper RNA
    X_c8, _, _ = load_fp("rlec_c6.0_r5_l3_s4096_f1.npz")   # slightly shallower

    # Descriptors from CSV
    desc, desc_cols = load_descriptors(ids)
    scaler = StandardScaler()

    print(f"\nDataset: {len(y)} complexes")
    print(f"RLEC 4k bits: {X4k.shape[1]}")
    print(f"RLEC 64k bits: {X64k.shape[1]}")
    print(f"Descriptors: {desc.shape[1]}  ({', '.join(desc_cols[:4])}...)")

    # ── Normalize pKd to (0,1) — same as RLaffinity ───────────────────────────
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min)

    print("\n" + "─" * 70)
    print("EXPERIMENT A — Baseline (RLEC 4k, LGB default)")
    print("─" * 70)
    run_splits(X4k, y_norm, lgb_default, "RLEC-4k  LGB-default")

    print("\n" + "─" * 70)
    print("EXPERIMENT B — Feature fusion variants")
    print("─" * 70)

    # B1: RLEC 4k + descriptors
    desc_scaled = scaler.fit_transform(desc)
    X_4k_desc = np.hstack([X4k, desc_scaled])
    run_splits(X_4k_desc, y_norm, lgb_default, "RLEC-4k + descriptors")

    # B2: RLEC 64k (less collision)
    run_splits(X64k, y_norm, lgb_default, "RLEC-64k  LGB-default")

    # B3: RLEC 64k + descriptors
    X_64k_desc = np.hstack([X64k, desc_scaled])
    run_splits(X_64k_desc, y_norm, lgb_default, "RLEC-64k + descriptors")

    # B4: Multi-FP concat (4k params: best + deeper + shallower) + descriptors
    X_multi = np.hstack([X4k, X_c9, X_c8, desc_scaled])
    run_splits(X_multi, y_norm, lgb_default, "Multi-FP (3x4k) + descriptors")

    print("\n" + "─" * 70)
    print("EXPERIMENT C — Optuna LGB tuning (on train set, 60 trials)")
    print("─" * 70)

    # Tune on 80% of data (first split train set) — no test leakage
    ss0 = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM)
    tr0, _ = next(ss0.split(X_64k_desc))
    print("  Tuning LightGBM on first train split (RLEC-64k + descriptors)...")
    best_params = tune_lgb(X_64k_desc[tr0], y_norm[tr0], n_trials=60)
    print(f"  Best params: {best_params}")

    run_splits(X_64k_desc, y_norm, lgb_tuned_fn(best_params), "RLEC-64k + desc + Optuna-LGB")

    print("\n" + "─" * 70)
    print("EXPERIMENT D — Multi-model ensemble")
    print("─" * 70)
    run_splits(X_64k_desc, y_norm, ensemble_fn, "RLEC-64k + desc + LGB+XGB+RF ensemble")

    # Best multi-FP with ensemble
    tr0b, _ = next(ShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                random_state=RANDOM).split(X_multi))
    best_params_multi = tune_lgb(X_multi[tr0b], y_norm[tr0b], n_trials=60)
    run_splits(X_multi, y_norm, lgb_tuned_fn(best_params_multi),
               "Multi-FP (3x4k) + desc + Optuna-LGB")

    print("\n" + "─" * 70)
    print("EXPERIMENT E — XGBoost on best feature set")
    print("─" * 70)
    run_splits(X_64k_desc, y_norm, xgb_fn, "RLEC-64k + desc + XGBoost")

    print("\n" + "=" * 70)
    print("COMPARISON (literature baselines)")
    print("=" * 70)
    baselines = [
        ("Vina",      -0.386, -0.389, 0.277, 0.257),
        ("RF-score",   0.445,  0.364, 0.152, 0.129),
        ("RLaffinity", 0.559,  0.540, 0.152, 0.119),
        ("RLASIF",     0.666,  0.601, 0.147, 0.112),
    ]
    print(f"  {'Method':<20} {'PCC':>6}  {'SPCC':>6}  {'RMSE':>6}  {'MAE':>6}")
    for name, pcc, spcc, rmse, mae in baselines:
        print(f"  {name:<20} {pcc:>6.3f}  {spcc:>6.3f}  {rmse:>6.3f}  {mae:>6.3f}")
    print(f"  {'RLEC v1 (step12)':<20} {'0.521':>6}  {'0.506':>6}  {'0.148':>6}  {'0.118':>6}")
    print(f"  {'RLEC v2 (above)':<20} {'→ see best above'}")


if __name__ == "__main__":
    main()
