"""
STEP 12 — Comparison with RLaffinity / RLASIF Protocol
RLEC on PDBbind NL2020 (143 complexes), same evaluation as Sun & Gao 2024

Protocol mirrors RLaffinity (btae155):
  - pKd min-max normalized globally to (0, 1)
  - 10 random 80/20 train-test splits (random_state seeds 0..9)
  - Metrics: PCC, SPCC, RMSE, MAE on normalized values
  - LightGBM (best model from step07c)

Comparison targets:
  RLaffinity : PCC=0.559  SPCC=0.540  RMSE=0.152  MAE=0.119  (144 complexes, 13 test)
  RLASIF     : PCC=0.666  SPCC=0.601  RMSE=0.147  AE=0.112   (117 complexes, 11 test)
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import ShuffleSplit
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / "data" / "features"
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

FEAT_FILE = FEAT_DIR / "rlec_c6.0_r6_l3_s4096_f1.npz"   # best params

N_SPLITS  = 10
TEST_SIZE = 0.20    # 20% test ≈ 29 complexes — close to RLaffinity's 13/142


def metrics(y_true, y_pred):
    r,  _ = stats.pearsonr(y_true, y_pred)
    rs, _ = stats.spearmanr(y_true, y_pred)
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    return dict(pcc=round(float(r), 4), spcc=round(float(rs), 4),
                rmse=round(rmse, 4),    mae=round(mae, 4))


def main():
    print("=" * 60)
    print("STEP 12 — RLaffinity-protocol evaluation")
    print(f"Feature file : {FEAT_FILE.name}")
    print(f"Splits       : {N_SPLITS} random (test={TEST_SIZE*100:.0f}%)")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────
    d = np.load(FEAT_FILE)
    X   = d["X"].astype(np.float32)          # (143, 4096)
    y   = d["y"].astype(np.float64)          # raw pKd
    ids = d["ids"]

    print(f"\nDataset      : {len(y)} complexes")
    print(f"pKd range    : {y.min():.3f} – {y.max():.3f}")

    # ── Global min-max normalization (same as RLaffinity) ─────────────
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min)   # → (0, 1)
    print(f"Normalized   : {y_norm.min():.3f} – {y_norm.max():.3f}")

    # ── LightGBM params (from step07c) ────────────────────────────────
    lgb_params = dict(
        objective="regression", metric="rmse",
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=4, random_state=42, verbose=-1,
    )

    # ── 10 random splits ──────────────────────────────────────────────
    ss = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=0)
    split_results = []

    for i, (tr_idx, te_idx) in enumerate(ss.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y_norm[tr_idx], y_norm[te_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])

        y_pred = model.predict(X_te)
        m = metrics(y_te, y_pred)
        split_results.append(m)
        print(f"  Split {i+1:2d}: PCC={m['pcc']:.3f}  SPCC={m['spcc']:.3f}  "
              f"RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  (n_test={len(te_idx)})")

    # ── Summary ───────────────────────────────────────────────────────
    df = pd.DataFrame(split_results)
    mean = df.mean()
    std  = df.std()

    print("\n" + "=" * 60)
    print("RESULTS (mean ± std over 10 splits)")
    print("=" * 60)
    print(f"  PCC  : {mean['pcc']:.3f} ± {std['pcc']:.3f}")
    print(f"  SPCC : {mean['spcc']:.3f} ± {std['spcc']:.3f}")
    print(f"  RMSE : {mean['rmse']:.3f} ± {std['rmse']:.3f}")
    print(f"  MAE  : {mean['mae']:.3f} ± {std['mae']:.3f}")

    print("\n" + "─" * 60)
    print("COMPARISON TABLE (10-split mean, normalized pKd)")
    print("─" * 60)
    comparison = [
        ("Vina",       -0.386, -0.389, 0.277, 0.257),
        ("RF-score",    0.445,  0.364, 0.152, 0.129),
        ("RSAPred",     0.399,  0.221, 2.886, 2.205),
        ("RLaffinity",  0.559,  0.540, 0.152, 0.119),
        ("RLASIF",      0.666,  0.601, 0.147, 0.112),
        ("RLEC (ours)", mean['pcc'], mean['spcc'], mean['rmse'], mean['mae']),
    ]
    print(f"  {'Method':<16} {'PCC':>6}  {'SPCC':>6}  {'RMSE':>6}  {'MAE':>6}")
    print(f"  {'-'*16} {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}")
    for name, pcc, spcc, rmse, mae in comparison:
        marker = " ←" if name == "RLEC (ours)" else ""
        print(f"  {name:<16} {pcc:>6.3f}  {spcc:>6.3f}  {rmse:>6.3f}  {mae:>6.3f}{marker}")

    # ── Save ──────────────────────────────────────────────────────────
    out = dict(
        protocol="RLaffinity (10 random 80/20 splits, normalized pKd)",
        n_complexes=int(len(y)),
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        mean=mean.to_dict(),
        std=std.to_dict(),
        per_split=split_results,
    )
    out_path = RES_DIR / "step12_rlaffinity_protocol.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
