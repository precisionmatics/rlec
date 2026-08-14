"""
STEP 05 — ML Model Training: Full Grid  (MAX CPU VERSION)
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Uses ALL 20 cores via ProcessPoolExecutor.
Results streamed to CSV as each file completes — no data loss on interrupt.
Resume support: skips already-completed combos automatically.

Optimisations:
  - 20 workers, each loads + trains independently (no shared state)
  - RF n_jobs=1  (20-way outer parallelism already saturates all cores)
  - NN: adam solver, (100,100) hidden layers, early stopping — ~5x faster than lbfgs
  - Files sorted small→large FP size: workers warm up on cheap jobs
  - CSV written + flushed after every completed future (streaming)
  - numpy/sklearn imports inside worker to avoid pickling overhead
"""

import sys, os, json, re, csv, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"      # prevent numpy from spawning extra threads
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / "data" / "features"
RES_DIR  = ROOT / "results"
LOG_DIR  = ROOT / "logs"
DATA_CSV = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")
RES_DIR.mkdir(exist_ok=True)

N_WORKERS = 20          # all physical cores
CSV_FLUSH = 10          # flush CSV every N completed futures

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── Parse filename ────────────────────────────────────────────────────────────
_PAT = re.compile(r"rlec_c([\d.]+)_r(\d+)_l(\d+)_s(\d+)_f(\d+)\.npz")

def parse_fname(fname):
    m = _PAT.match(fname)
    if not m:
        return None
    return dict(cutoff=float(m.group(1)), rna_depth=int(m.group(2)),
                lig_depth=int(m.group(3)), fp_size=int(m.group(4)),
                feat_set=int(m.group(5)))


# ── Worker function (runs in subprocess) ──────────────────────────────────────
def train_one_file(npz_path_str, train_idx, test_idx):
    """
    Load one npz FP matrix, train linear + RF + NN, return list of result dicts.
    All heavy imports done inside this function — avoids pickling sklearn objects.
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    from scipy import stats
    from sklearn.linear_model import SGDRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    npz_path = Path(npz_path_str)
    params   = parse_fname(npz_path.name)
    if params is None:
        return []

    try:
        data = np.load(npz_path)
        X    = data["X"]
        y    = data["y"]
    except Exception:
        return []

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    def metrics(y_true, y_pred):
        if len(y_true) < 3:
            return np.nan, np.nan, np.nan
        r, _ = stats.pearsonr(y_true, y_pred)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        sd   = float(np.std(y_true - y_pred))
        return round(float(r), 4), round(rmse, 4), round(sd, 4)

    results = []

    # ── Linear SGD ──
    try:
        m = SGDRegressor(loss="huber", penalty="elasticnet", l1_ratio=0.5,
                         epsilon=0.1, max_iter=1000, tol=1e-3, random_state=42)
        m.fit(X_tr, y_tr)
        r, rmse, sd = metrics(y_te, m.predict(X_te))
        results.append({**params, "model": "linear", "pearson_r": r, "rmse": rmse, "sd": sd})
    except Exception:
        results.append({**params, "model": "linear", "pearson_r": np.nan, "rmse": np.nan, "sd": np.nan})

    # ── Random Forest ──
    try:
        m = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42)
        m.fit(X_tr, y_tr)
        r, rmse, sd = metrics(y_te, m.predict(X_te))
        results.append({**params, "model": "rf", "pearson_r": r, "rmse": rmse, "sd": sd})
    except Exception:
        results.append({**params, "model": "rf", "pearson_r": np.nan, "rmse": np.nan, "sd": np.nan})

    # ── Neural Network (adam — fast) ──
    try:
        scaler = StandardScaler(with_mean=False)
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        m = MLPRegressor(hidden_layer_sizes=(100, 100), activation="relu",
                         solver="adam", max_iter=300,
                         early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=10, random_state=42)
        m.fit(X_tr_s, y_tr)
        r, rmse, sd = metrics(y_te, m.predict(X_te_s))
        results.append({**params, "model": "nn", "pearson_r": r, "rmse": rmse, "sd": sd})
    except Exception:
        results.append({**params, "model": "nn", "pearson_r": np.nan, "rmse": np.nan, "sd": np.nan})

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info(f"RLEC Step 05 — Full Grid (ALL {N_WORKERS} CORES)")
    log.info("=" * 60)

    # ── Split ──
    df_meta  = pd.read_csv(DATA_CSV)[["pdb", "subtype"]]
    sample   = np.load(sorted(FEAT_DIR.glob("*.npz"))[0])
    pdb_ids  = sample["ids"]
    subtype_map = dict(zip(df_meta["pdb"], df_meta["subtype"]))
    subtypes    = np.array([subtype_map.get(p, "unknown") for p in pdb_ids])

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(np.zeros(len(pdb_ids)), subtypes))
    log.info(f"Train: {len(train_idx)}   Test: {len(test_idx)}")

    # ── Resume: find already-done combos ──
    out_csv = RES_DIR / "step05_grid_results.csv"
    done_combos = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_csv)
            for _, row in existing.iterrows():
                done_combos.add((row["cutoff"], int(row["rna_depth"]),
                                 int(row["lig_depth"]), int(row["fp_size"]),
                                 int(row["feat_set"])))
            log.info(f"Resume: {len(done_combos)} combos already done")
        except Exception:
            pass

    # ── Build todo list sorted small FP → large FP ──
    all_npz = sorted(FEAT_DIR.glob("*.npz"),
                     key=lambda f: (parse_fname(f.name) or {}).get("fp_size", 0))
    todo = []
    for f in all_npz:
        p = parse_fname(f.name)
        if p and (p["cutoff"], p["rna_depth"], p["lig_depth"],
                  p["fp_size"], p["feat_set"]) not in done_combos:
            todo.append(f)

    total  = len(todo)
    log.info(f"Files to process: {total} / {len(all_npz)}")
    log.info(f"Total model fits: {total * 3}")
    log.info(f"Workers: {N_WORKERS}")

    if total == 0:
        log.info("All done already.")
        return 0

    # ── Open CSV for streaming writes ──
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    fieldnames   = ["cutoff","rna_depth","lig_depth","fp_size","feat_set",
                    "model","pearson_r","rmse","sd"]
    csv_fh  = open(out_csv, "a", newline="", buffering=1)
    writer  = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    # ── Submit all tasks ──
    t_start   = datetime.now()
    completed = 0
    pending   = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(train_one_file, str(f), train_idx, test_idx): f
            for f in todo
        }
        pending = len(futures)

        for fut in as_completed(futures):
            try:
                rows = fut.result()
                for r in rows:
                    writer.writerow(r)
            except Exception as e:
                log.warning(f"Worker error: {e}")

            completed += 1
            if completed % CSV_FLUSH == 0:
                csv_fh.flush()

            if completed % 200 == 0 or completed == pending:
                elapsed = (datetime.now() - t_start).total_seconds()
                rate    = completed / elapsed
                eta_min = (pending - completed) / rate / 60 if rate > 0 else 0
                pct     = 100 * completed / pending
                log.info(f"  {completed}/{pending}  ({pct:.1f}%)  "
                         f"rate={rate:.1f} files/s  ETA={eta_min:.1f} min")

    csv_fh.flush()
    csv_fh.close()

    # ── Final summary ──
    elapsed = (datetime.now() - t_start).total_seconds()
    results_df = pd.read_csv(out_csv)
    log.info(f"\n{'=' * 60}")
    log.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"Total rows: {len(results_df)}")
    log.info(f"NaN results: {results_df['pearson_r'].isna().sum()}")

    log.info(f"\nTop 10 by Pearson r:")
    top10 = results_df.nlargest(10,"pearson_r")[
        ["model","cutoff","rna_depth","lig_depth","fp_size","feat_set","pearson_r","rmse"]]
    log.info(f"\n{top10.to_string(index=False)}")

    log.info(f"\nBest per model:")
    for mdl in ["linear","rf","nn"]:
        sub = results_df[results_df["model"] == mdl]
        if sub.empty: continue
        b = sub.nlargest(1,"pearson_r").iloc[0]
        log.info(f"  {mdl:8s}: r={b['pearson_r']:.4f}  cutoff={b['cutoff']}  "
                 f"rna_d={b['rna_depth']}  lig_d={b['lig_depth']}  "
                 f"fp={b['fp_size']}  feat={b['feat_set']}")

    run_log = {"timestamp": t_start.isoformat(), "elapsed_sec": elapsed,
               "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
               "n_files": total, "n_results": len(results_df),
               "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}
    with open(LOG_DIR / "step05_training.json", "w") as f:
        json.dump(run_log, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
