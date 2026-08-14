"""
STEP 11 — Feature Enhancement: Multi-cutoff Stack, Ligand Descriptors, Sequence Neighborhood
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Three enhancements beyond baseline RLEC feat1 (LOOCV r=0.7097):

  A. Multi-cutoff FP stack
     Concatenate RLEC feat1 FPs at c=4.5, 5.5, 6.0 Å → 3×4096 = 12288-dim.
     Uses existing npz files (zero extra computation). Adds spatial resolution.

  B. Ligand global descriptors appended
     RDKit: MW, LogP, TPSA, nHBD, nHBA, nRotBonds → 6 normalized scalars
     appended to RLEC feat1 FP → 4102-dim.

  C. Sequence neighborhood hash (feat_set=4, novel RNA contribution)
     Extends depth-0 RNA atom invariant to include nuc_type of 5′ and 3′
     neighboring residues (from (chain, resnum) ± 1 in sorted sequence).
     Encodes local RNA sequence context — PLEC has no protein analogue.

  D. Combined: multi-cutoff stack of seq-neighborhood FP + global descriptors
     Best of all: 3×4096 (feat4 at 3 cutoffs) + 6 descriptors = 12294-dim

All compared with LOOCV LightGBM Optuna (n=143, pooled Pearson r).

Outputs:
  results/step11_enhancement_results.json
  logs/step11_enhancement.json
  data/features/rlec_c6.0_r6_l3_s4096_f4.npz  (seq-neighborhood FP)
  data/features/rlec_c5.5_r6_l3_s4096_f4.npz
  data/features/rlec_c4.5_r6_l3_s4096_f4.npz
"""

import sys, os, json, logging, warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from itertools import zip_longest
from collections import defaultdict
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

ROOT        = Path(__file__).parent.parent
CONTACT_DIR = ROOT / "data" / "contacts"
FEAT_DIR    = ROOT / "data" / "features"
RES_DIR     = ROOT / "results"
LOG_DIR     = ROOT / "logs"
DATA_CSV    = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")

BEST_CUTOFF  = 6.0
BEST_RNA_D   = 6
BEST_LIG_D   = 3
BEST_FP_SIZE = 4096
BEST_FEAT    = 1

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_lgb_params():
    with open(RES_DIR / "step07d_optuna_results.json") as f:
        return json.load(f)["results"]["lgb_optuna"]["best_params"]


def loocv_predict(X, y, label=""):
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
        log.info(f"  LOOCV {label}: r={r:.4f}  RMSE={rmse:.4f}  shape={X.shape}")
    return y_pred, round(r, 4), round(rmse, 4)


def load_npz(cutoff, rna_d, lig_d, fp_size, feat):
    name = f"rlec_c{cutoff:.1f}_r{rna_d}_l{lig_d}_s{fp_size}_f{feat}.npz"
    path = FEAT_DIR / name
    if not path.exists():
        return None, None, None
    d = np.load(path)
    return d["X"], d["y"], d["ids"]


def to_python(obj):
    if isinstance(obj, dict):           return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, bool):           return obj
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# RNA ECFP hashing (from step04, extended with seq-neighborhood feat_set=4)
# ─────────────────────────────────────────────────────────────────────────────

def _rna_atom_invariant(atom, feat_set, neighbor_nuc=None):
    """
    Depth-0 hash for an RNA atom.
    feat_set 4: feat1 (nuc_type) + 5'/3' neighbor nuc_types from sequence.
    neighbor_nuc: (nuc_5prime, nuc_3prime) tuple, or None for non-4 feat_sets.
    """
    base = (
        atom["atomic_num"],
        atom["formal_charge"],
        atom["n_heavy_nbrs"],
        atom["is_aromatic"],
        atom["in_ring"],
    )
    if feat_set == 0:
        features = base
    elif feat_set == 1:
        features = base + (atom["nuc_type"],)
    elif feat_set == 2:
        features = base + (atom["pbs"],)
    elif feat_set == 3:
        features = base + (atom["nuc_type"], atom["pbs"])
    else:  # feat_set == 4: sequence neighborhood
        nuc_5, nuc_3 = neighbor_nuc if neighbor_nuc is not None else (-1, -1)
        features = base + (atom["nuc_type"], nuc_5, nuc_3)
    return hash(features) & 0xFFFFFFFF


def build_seq_neighbor_map(rna_atoms):
    """
    Returns {atom_idx: (nuc_type_5prime, nuc_type_3prime)}.
    Uses (chain, resnum) to identify neighboring nucleotides in sequence.
    -1 encodes terminal (no neighbor).
    """
    res_nuc  = {}   # (chain, resnum) → nuc_type
    atom_res = {}   # atom_idx → (chain, resnum)
    for a in rna_atoms:
        key = (a["chain"], a["resnum"])
        res_nuc[key]     = a["nuc_type"]
        atom_res[a["idx"]] = key

    chain_resnums = defaultdict(list)
    for chain, resnum in res_nuc:
        chain_resnums[chain].append(resnum)
    for ch in chain_resnums:
        chain_resnums[ch].sort()

    res_pos = {}   # (chain, resnum) → sorted position within chain
    for ch, rns in chain_resnums.items():
        for i, rn in enumerate(rns):
            res_pos[(ch, rn)] = i

    neighbor_map = {}
    for a in rna_atoms:
        key   = atom_res[a["idx"]]
        ch, _ = key
        pos   = res_pos[key]
        rns   = chain_resnums[ch]

        nuc_5 = res_nuc.get((ch, rns[pos - 1]), -1) if pos > 0 else -1
        nuc_3 = res_nuc.get((ch, rns[pos + 1]), -1) if pos < len(rns) - 1 else -1
        neighbor_map[a["idx"]] = (nuc_5, nuc_3)

    return neighbor_map


def compute_rna_ecfp_hashes(atoms, graph, max_depth, feat_set, neighbor_map=None):
    """Morgan-style ECFP hashes for all RNA atoms up to max_depth."""
    current = {}
    for a in atoms:
        nb = neighbor_map.get(a["idx"]) if neighbor_map else None
        current[a["idx"]] = _rna_atom_invariant(a, feat_set, nb)

    all_hashes = {a["idx"]: [current[a["idx"]]] for a in atoms}

    for depth in range(1, max_depth + 1):
        new_hashes = {}
        for atom in atoms:
            idx       = atom["idx"]
            nbr_hashes = tuple(sorted(current[j] for j in graph.get(idx, [])))
            h = hash((depth, current[idx]) + nbr_hashes) & 0xFFFFFFFF
            new_hashes[idx] = h
        current = new_hashes
        for idx, h in current.items():
            all_hashes[idx].append(h)

    return all_hashes


def compute_ligand_ecfp_hashes(mol, max_depth):
    """Per-atom Morgan hashes at each depth 0..max_depth using RDKit bitInfo."""
    result = {atom.GetIdx(): [] for atom in mol.GetAtoms()
              if atom.GetAtomicNum() != 1}
    for depth in range(max_depth + 1):
        bi = {}
        AllChem.GetMorganFingerprint(mol, depth, bitInfo=bi)
        atom_to_hash = {}
        for bit_hash, atom_rad_list in bi.items():
            for atom_idx, radius in atom_rad_list:
                if radius == depth:
                    atom_to_hash[atom_idx] = bit_hash
        for atom_idx in result:
            h = atom_to_hash.get(atom_idx, result[atom_idx][0] if result[atom_idx] else 0)
            result[atom_idx].append(h)
    return result


def rlec_bits_for_complex(rna_hashes, lig_hashes, contacts, rna_depth, lig_depth):
    """Core RLEC pairing: zip_longest(rna_env, lig_env) → hash pairs."""
    bits = []
    for rna_idx, lig_idx, _ in contacts:
        rna_env = rna_hashes.get(rna_idx)
        lig_env = lig_hashes.get(lig_idx)
        if rna_env is None or lig_env is None:
            continue
        rna_env  = rna_env[:rna_depth + 1]
        lig_env  = lig_env[:lig_depth + 1]
        rna_fill = rna_env[-1] if rna_env else 0
        lig_fill = lig_env[-1] if lig_env else 0
        for r_h, l_h in zip_longest(rna_env, lig_env, fillvalue=None):
            r_h = r_h if r_h is not None else rna_fill
            l_h = l_h if l_h is not None else lig_fill
            bits.append(hash((r_h, l_h)) & 0xFFFFFFFF)
    return bits


def fold_bits(bits, fp_size):
    vec = np.zeros(fp_size, dtype=np.float32)
    for b in bits:
        vec[b % fp_size] += 1
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Build feat4 (sequence-neighborhood) FP for given cutoff
# ─────────────────────────────────────────────────────────────────────────────

def build_feat4_fp(complexes, lig_cache, cutoff, rna_d, lig_d, fp_size):
    """Build RLEC feat4 (seq-neighborhood) for all complexes at given cutoff."""
    n = len(complexes)
    X = np.zeros((n, fp_size), dtype=np.float32)
    for ci, cdata in enumerate(complexes):
        neighbor_map = build_seq_neighbor_map(cdata["rna_atoms"])
        rna_hashes   = compute_rna_ecfp_hashes(
            cdata["rna_atoms"], cdata["rna_bond_graph"],
            rna_d, feat_set=4, neighbor_map=neighbor_map
        )
        contacts = cdata["contacts"].get(cutoff, [])
        if not contacts:
            continue
        bits = rlec_bits_for_complex(rna_hashes, lig_cache[ci], contacts, rna_d, lig_d)
        if bits:
            X[ci] = fold_bits(bits, fp_size)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Ligand global descriptors
# ─────────────────────────────────────────────────────────────────────────────

def compute_lig_descriptors(complexes, pdb_order):
    """
    RDKit: MW, LogP, TPSA, nHBD, nHBA, nRotBonds for each complex.
    Returns StandardScaler-normalized (n, 6) matrix ordered by pdb_order.
    """
    pdb_to_mol = {c["pdb_id"]: c["rdkit_mol"] for c in complexes}
    rows = []
    for pdb_id in pdb_order:
        mol = pdb_to_mol.get(pdb_id)
        if mol is None:
            rows.append([0.0] * 6)
            continue
        rows.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            float(Descriptors.NumHDonors(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
        ])
    X_desc = np.array(rows, dtype=np.float32)
    X_desc = StandardScaler().fit_transform(X_desc)
    return X_desc.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("RLEC Step 11 — Feature Enhancement")
    log.info("=" * 60)
    t_start = datetime.now()

    # ── Load complexes ──
    log.info("Loading complexes...")
    pkl_files  = sorted(CONTACT_DIR.glob("*_contacts.pkl"))
    complexes  = []
    failed     = []
    for f in pkl_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        pdb_id   = d["pdb_id"]
        sdf_path = Path(f"/home/stalin/Desktop/PDFL-RNA/NA-L/{pdb_id}/{pdb_id}_ligand.sdf")
        mol = None
        if sdf_path.exists():
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
            mol   = next((m for m in suppl if m is not None), None)
            if mol is None:
                suppl2 = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=False)
                mol    = next((m for m in suppl2 if m is not None), None)
        if mol is None:
            failed.append(pdb_id)
            continue
        d["rdkit_mol"] = mol
        complexes.append(d)
    log.info(f"Loaded: {len(complexes)}  Failed: {len(failed)}")

    pdb_ids  = np.array([c["pdb_id"] for c in complexes])
    y        = np.array([c["pKd"]    for c in complexes], dtype=np.float32)

    # ── Pre-compute ligand ECFP hashes (shared across enhancements) ──
    log.info("Pre-computing ligand ECFP hashes...")
    lig_cache = []
    for cdata in complexes:
        lig_cache.append(compute_ligand_ecfp_hashes(cdata["rdkit_mol"], BEST_LIG_D))

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # BASELINE: RLEC feat1 c6.0 (confirmed best from steps 07d/08/10)
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- BASELINE: RLEC feat1 c6.0 (LOOCV r=0.7097 from step08) ---")
    X_base, y_base, ids_base = load_npz(6.0, BEST_RNA_D, BEST_LIG_D, BEST_FP_SIZE, 1)
    assert np.all(ids_base == pdb_ids), "ID order mismatch in baseline npz"
    _, r_base, rmse_base = loocv_predict(X_base, y, "feat1 c6.0 (baseline)")
    results["baseline_feat1_c6"] = {"loocv_r": r_base, "loocv_rmse": rmse_base,
                                     "shape": list(X_base.shape)}

    # ─────────────────────────────────────────────────────────────────────────
    # ENHANCEMENT A: Multi-cutoff FP stack (feat1 at c4.5 + c5.5 + c6.0)
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- ENHANCEMENT A: Multi-cutoff stack (feat1 c4.5+c5.5+c6.0) ---")
    stacks = []
    for c in [4.5, 5.5, 6.0]:
        Xc, _, idsc = load_npz(c, BEST_RNA_D, BEST_LIG_D, BEST_FP_SIZE, 1)
        assert Xc is not None, f"Missing npz for c={c}"
        assert np.all(idsc == pdb_ids), f"ID mismatch at c={c}"
        stacks.append(Xc)
    X_stack = np.hstack(stacks).astype(np.float32)
    log.info(f"  Stacked shape: {X_stack.shape}")
    _, r_stack, rmse_stack = loocv_predict(X_stack, y, "multi-cutoff stack")
    results["enhancement_A_multicutoff"] = {
        "loocv_r": r_stack, "loocv_rmse": rmse_stack,
        "shape": list(X_stack.shape), "cutoffs": [4.5, 5.5, 6.0],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # ENHANCEMENT B: Ligand global descriptors appended to feat1 c6.0
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- ENHANCEMENT B: RLEC feat1 + ligand global descriptors (6 RDKit) ---")
    X_desc = compute_lig_descriptors(complexes, pdb_ids)
    log.info(f"  Descriptor stats: {X_desc.mean(0).round(2)}")
    X_desc_only = X_desc.copy()
    X_with_desc = np.hstack([X_base, X_desc]).astype(np.float32)
    log.info(f"  Combined shape: {X_with_desc.shape}")

    # Descriptors alone (sanity)
    _, r_desc_only, rmse_desc_only = loocv_predict(X_desc_only, y, "descriptors only (6)")
    results["enhancement_B_desc_only"] = {
        "loocv_r": r_desc_only, "loocv_rmse": rmse_desc_only,
        "shape": list(X_desc_only.shape),
        "descriptors": ["MW", "LogP", "TPSA", "nHBD", "nHBA", "nRotBonds"],
    }
    _, r_with_desc, rmse_with_desc = loocv_predict(X_with_desc, y, "feat1 + descriptors")
    results["enhancement_B_feat1_plus_desc"] = {
        "loocv_r": r_with_desc, "loocv_rmse": rmse_with_desc,
        "shape": list(X_with_desc.shape),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # ENHANCEMENT C: Sequence neighborhood (feat4) — rebuild FP from pkl
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- ENHANCEMENT C: Sequence neighborhood feat4 (5'/3' nuc_type in hash) ---")
    feat4_cutoffs = [4.5, 5.5, 6.0]
    feat4_fps     = {}
    for c in feat4_cutoffs:
        npz_path = FEAT_DIR / f"rlec_c{c:.1f}_r{BEST_RNA_D}_l{BEST_LIG_D}_s{BEST_FP_SIZE}_f4.npz"
        if npz_path.exists():
            log.info(f"  Loading cached feat4 npz at c={c}...")
            d = np.load(npz_path)
            feat4_fps[c] = d["X"]
            assert np.all(d["ids"] == pdb_ids), f"ID mismatch in feat4 npz c={c}"
        else:
            log.info(f"  Building feat4 FP at c={c} (this may take ~30s)...")
            X_f4 = build_feat4_fp(complexes, lig_cache, c,
                                   BEST_RNA_D, BEST_LIG_D, BEST_FP_SIZE)
            np.savez_compressed(npz_path, X=X_f4, y=y, ids=pdb_ids)
            log.info(f"  Saved: {npz_path.name}")
            feat4_fps[c] = X_f4

    # feat4 at c6.0 alone
    X_f4_60 = feat4_fps[6.0]
    log.info(f"  feat4 c6.0 shape: {X_f4_60.shape}  nonzero: {(X_f4_60.sum(1)>0).sum()}")
    _, r_f4, rmse_f4 = loocv_predict(X_f4_60, y, "feat4 c6.0 (seq-neighborhood)")
    results["enhancement_C_feat4_c6"] = {
        "loocv_r": r_f4, "loocv_rmse": rmse_f4,
        "shape": list(X_f4_60.shape),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # COMBINATIONS
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n--- COMBINATIONS ---")

    # A + B: multi-cutoff stack + global descriptors
    X_AB = np.hstack([X_stack, X_desc]).astype(np.float32)
    _, r_AB, rmse_AB = loocv_predict(X_AB, y, "A+B: multicutoff + descriptors")
    results["combo_AB_multicutoff_desc"] = {
        "loocv_r": r_AB, "loocv_rmse": rmse_AB, "shape": list(X_AB.shape),
    }

    # C_stack: feat4 multi-cutoff stack (4.5+5.5+6.0)
    X_C_stack = np.hstack([feat4_fps[c] for c in feat4_cutoffs]).astype(np.float32)
    _, r_C_stack, rmse_C_stack = loocv_predict(X_C_stack, y, "feat4 multi-cutoff stack")
    results["combo_C_feat4_multicutoff"] = {
        "loocv_r": r_C_stack, "loocv_rmse": rmse_C_stack,
        "shape": list(X_C_stack.shape),
    }

    # C_stack + B: feat4 multi-cutoff + global descriptors
    X_CB = np.hstack([X_C_stack, X_desc]).astype(np.float32)
    _, r_CB, rmse_CB = loocv_predict(X_CB, y, "feat4 multicutoff + descriptors")
    results["combo_CB_feat4_multicutoff_desc"] = {
        "loocv_r": r_CB, "loocv_rmse": rmse_CB, "shape": list(X_CB.shape),
    }

    # feat1 + feat4 at c6.0 (direct complement)
    X_f1f4 = np.hstack([X_base, X_f4_60]).astype(np.float32)
    _, r_f1f4, rmse_f1f4 = loocv_predict(X_f1f4, y, "feat1 + feat4 c6.0")
    results["combo_feat1_feat4_c6"] = {
        "loocv_r": r_f1f4, "loocv_rmse": rmse_f1f4, "shape": list(X_f1f4.shape),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - t_start).total_seconds()
    out = {
        "timestamp"   : t_start.isoformat(),
        "elapsed_sec" : elapsed,
        "model"       : "LightGBM (Optuna-tuned)",
        "eval"        : "LOOCV n=143, pooled Pearson r",
        "baseline_r"  : 0.7097,
        "results"     : results,
    }

    out_path = RES_DIR / "step11_enhancement_results.json"
    with open(out_path, "w") as f:
        json.dump(to_python(out), f, indent=2)
    with open(LOG_DIR / "step11_enhancement.json", "w") as f:
        json.dump(to_python(out), f, indent=2)

    log.info(f"\n{'='*60}")
    log.info("ENHANCEMENT COMPARISON TABLE")
    log.info(f"{'Baseline RLEC feat1 c6.0':<45}: r=0.7097  (from step08)")
    ordered = [
        ("enhancement_A_multicutoff",        "A: multi-cutoff stack (4.5+5.5+6.0)"),
        ("enhancement_B_desc_only",           "B0: ligand descriptors only (6)"),
        ("enhancement_B_feat1_plus_desc",     "B: feat1 + ligand descriptors"),
        ("enhancement_C_feat4_c6",            "C: feat4 seq-neighborhood c6.0"),
        ("combo_AB_multicutoff_desc",         "A+B: multicutoff + descriptors"),
        ("combo_C_feat4_multicutoff",         "C-stack: feat4 multicutoff"),
        ("combo_CB_feat4_multicutoff_desc",   "C-stack+B: feat4 multicutoff + desc"),
        ("combo_feat1_feat4_c6",              "feat1+feat4 c6.0"),
    ]
    best_r  = 0.7097
    best_key = "baseline"
    for key, label in ordered:
        if key not in results:
            continue
        v   = results[key]
        gap = v["loocv_r"] - 0.7097
        marker = " ← NEW BEST" if v["loocv_r"] > best_r else ""
        if v["loocv_r"] > best_r:
            best_r   = v["loocv_r"]
            best_key = key
        log.info(f"  {label:<45}: r={v['loocv_r']:.4f}  Δ={gap:+.4f}  RMSE={v['loocv_rmse']:.4f}{marker}")

    log.info(f"\n  BEST OVERALL: r={best_r:.4f}  ({best_key})")
    log.info(f"\nDone in {elapsed:.1f}s  |  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
