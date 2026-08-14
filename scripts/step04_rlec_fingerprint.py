"""
STEP 04 — RLEC Fingerprint Computation
RLEC: RNA-Ligand Extended Connectivity Fingerprint

Core RLEC algorithm — mirrors PLEC (Wójcikowski et al., Bioinformatics 2019)
but adapted for RNA:

  rlec_bits = []
  for rna_atom, lig_atom in contacts(rna, ligand, cutoff):
      rna_ecfp = ECFP_hashes(rna_atom, depth=rna_depth)   # depth+1 hashes
      lig_ecfp = ECFP_hashes(lig_atom, depth=lig_depth)   # depth+1 hashes
      for envs_pair in zip_longest(rna_ecfp, lig_ecfp):
          rlec_bits.append(hash(envs_pair))

RNA ECFP hashes: computed manually via Morgan-style iteration on the RNA bond
graph, using 4 feature sets (feat0..feat3).

Ligand ECFP hashes: computed via RDKit GetMorganFingerprint bit info, which
gives per-atom per-radius hashes natively.

Sweeps ALL parameter combinations:
  cutoff    : 7 values  (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 Å)
  rna_depth : 6 values  (1 – 6)
  lig_depth : 6 values  (1 – 6)
  fp_size   : 4 values  (4096, 16384, 32768, 65536)
  feat_set  : 4 values  (0, 1, 2, 3)
  → 7×6×6×4×4 = 4032 FP matrices

Outputs:
  data/features/{cutoff}_{rna_depth}_{lig_depth}_{fp_size}_{feat}.npz
    → 'X'  : float32 sparse-friendly matrix (n_complexes, fp_size)
    → 'y'  : float32 pKd vector (n_complexes,)
    → 'ids': str array of PDB IDs

  logs/step04_fingerprint.json  — run summary and timing
"""

import sys
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import zip_longest
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CONTACT_DIR = ROOT / "data" / "contacts"
FEAT_DIR    = ROOT / "data" / "features"
LOG_DIR     = ROOT / "logs"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameter grid ─────────────────────────────────────────────────────────────
CUTOFFS    = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
RNA_DEPTHS = [1, 2, 3, 4, 5, 6]
LIG_DEPTHS = [1, 2, 3, 4, 5, 6]
FP_SIZES   = [4096, 16384, 32768, 65536]
FEAT_SETS  = [0, 1, 2, 3]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RNA ECFP hashing
# ─────────────────────────────────────────────────────────────────────────────

def _rna_atom_invariant(atom: dict, feat_set: int) -> int:
    """Depth-0 hash for an RNA atom under the given feature set."""
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
    else:  # feat_set == 3
        features = base + (atom["nuc_type"], atom["pbs"])
    return hash(features) & 0xFFFFFFFF


def compute_rna_ecfp_hashes(atoms: list[dict],
                              graph: dict[int, list[int]],
                              max_depth: int,
                              feat_set: int) -> dict[int, list[int]]:
    """
    Morgan-style ECFP hashes for all RNA atoms up to max_depth.
    Returns {atom_idx: [hash_d0, hash_d1, ..., hash_d(max_depth)]}
    """
    # Build idx → position mapping for fast lookup
    idx_to_pos = {a["idx"]: i for i, a in enumerate(atoms)}

    # Depth-0 invariants
    current = {a["idx"]: _rna_atom_invariant(a, feat_set) for a in atoms}

    # Per-atom hash list: starts with depth-0 hash
    all_hashes = {a["idx"]: [current[a["idx"]]] for a in atoms}

    for depth in range(1, max_depth + 1):
        new_hashes = {}
        for atom in atoms:
            idx = atom["idx"]
            nbr_hashes = tuple(sorted(current[j] for j in graph.get(idx, [])))
            h = hash((depth, current[idx]) + nbr_hashes) & 0xFFFFFFFF
            new_hashes[idx] = h
        current = new_hashes
        for idx, h in current.items():
            all_hashes[idx].append(h)

    return all_hashes   # {idx: [d0, d1, ..., d_max]}


# ─────────────────────────────────────────────────────────────────────────────
# Ligand ECFP hashing via RDKit
# ─────────────────────────────────────────────────────────────────────────────

def compute_ligand_ecfp_hashes(mol, max_depth: int) -> dict[int, list[int]]:
    """
    Per-atom Morgan hashes at each depth 0..max_depth using RDKit bitInfo.
    Returns {atom_idx: [hash_d0, hash_d1, ..., hash_d_max]}
    """
    result = {atom.GetIdx(): [] for atom in mol.GetAtoms()
              if atom.GetAtomicNum() != 1}

    for depth in range(max_depth + 1):
        bi = {}
        AllChem.GetMorganFingerprint(mol, depth, bitInfo=bi)
        # bi = {bit_hash: [(atom_idx, radius), ...]}
        # Invert: atom_idx → hash at this radius
        atom_to_hash = {}
        for bit_hash, atom_rad_list in bi.items():
            for atom_idx, radius in atom_rad_list:
                if radius == depth:
                    atom_to_hash[atom_idx] = bit_hash

        for atom_idx in result:
            # If no hash at this depth (atom not in fingerprint), use depth-0 hash
            h = atom_to_hash.get(atom_idx, result[atom_idx][0] if result[atom_idx] else 0)
            result[atom_idx].append(h)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RLEC fingerprint for one complex
# ─────────────────────────────────────────────────────────────────────────────

def rlec_bits_for_complex(rna_hashes: dict[int, list[int]],
                           lig_hashes: dict[int, list[int]],
                           contacts: list[tuple],
                           rna_depth: int,
                           lig_depth: int) -> list[int]:
    """
    Core RLEC algorithm:
      For each contact pair (rna_idx, lig_idx):
        zip_longest(rna_ecfp[rna_idx][:rna_depth+1],
                    lig_ecfp[lig_idx][:lig_depth+1],
                    fillvalue=last_element_of_shorter)
        → hash each pair → append to bits
    """
    bits = []
    for rna_idx, lig_idx, _ in contacts:
        rna_env = rna_hashes.get(rna_idx)
        lig_env = lig_hashes.get(lig_idx)
        if rna_env is None or lig_env is None:
            continue

        rna_env = rna_env[:rna_depth + 1]
        lig_env = lig_env[:lig_depth + 1]

        # zip_longest with last element of shorter as fillvalue
        rna_fill = rna_env[-1] if rna_env else 0
        lig_fill = lig_env[-1] if lig_env else 0

        for r_h, l_h in zip_longest(rna_env, lig_env,
                                     fillvalue=None):
            r_h = r_h if r_h is not None else rna_fill
            l_h = l_h if l_h is not None else lig_fill
            bits.append(hash((r_h, l_h)) & 0xFFFFFFFF)

    return bits


def fold_bits(bits: list[int], fp_size: int) -> np.ndarray:
    """Fold raw 32-bit hashes into a count vector of length fp_size."""
    vec = np.zeros(fp_size, dtype=np.float32)
    for b in bits:
        vec[b % fp_size] += 1
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Load all complex data
# ─────────────────────────────────────────────────────────────────────────────

def load_all_complexes() -> list[dict]:
    """Load all pkl files. Returns list of complex dicts with rdkit mol added."""
    pkl_files = sorted(CONTACT_DIR.glob("*_contacts.pkl"))
    log.info(f"Loading {len(pkl_files)} complexes...")

    complexes = []
    failed = []
    for pkl_path in tqdm(pkl_files, desc="Loading"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Load RDKit mol for ligand (needed for ligand ECFP hashes)
        sdf_path = None
        # Reconstruct SDF path from pdb_id
        pdb_id = data["pdb_id"]
        sdf_path = Path(f"/home/stalin/Desktop/PDFL-RNA/NA-L/{pdb_id}/{pdb_id}_ligand.sdf")

        mol = None
        if sdf_path.exists():
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
            mol = next((m for m in suppl if m is not None), None)
            if mol is None:
                suppl2 = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=False)
                mol = next((m for m in suppl2 if m is not None), None)

        if mol is None:
            failed.append(pdb_id)
            continue

        data["rdkit_mol"] = mol
        complexes.append(data)

    if failed:
        log.warning(f"Failed to load ligand mol for: {failed}")
    log.info(f"Loaded {len(complexes)} complexes successfully.")
    return complexes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("RLEC Step 04 — Fingerprint Computation")
    log.info(f"Grid: {len(CUTOFFS)} cutoffs × {len(RNA_DEPTHS)} rna_depths × "
             f"{len(LIG_DEPTHS)} lig_depths × {len(FP_SIZES)} fp_sizes × "
             f"{len(FEAT_SETS)} feat_sets = "
             f"{len(CUTOFFS)*len(RNA_DEPTHS)*len(LIG_DEPTHS)*len(FP_SIZES)*len(FEAT_SETS)} combinations")
    log.info("=" * 60)

    t_start = datetime.now()

    # 1. Load all complexes
    complexes = load_all_complexes()
    n = len(complexes)
    pdb_ids = np.array([c["pdb_id"] for c in complexes])
    y       = np.array([c["pKd"]    for c in complexes], dtype=np.float32)

    # 2. Pre-compute RNA ECFP hashes for all feat_sets and max depth (6)
    #    We compute at max_depth=6 once and slice for smaller depths
    MAX_RNA_DEPTH = max(RNA_DEPTHS)
    MAX_LIG_DEPTH = max(LIG_DEPTHS)

    log.info(f"\nPre-computing RNA ECFP hashes (max_depth={MAX_RNA_DEPTH}, "
             f"feat_sets={FEAT_SETS})...")

    # rna_ecfp_cache[feat_set][complex_idx] = {atom_idx: [h0, h1, ..., h6]}
    rna_ecfp_cache = {}
    for feat_set in FEAT_SETS:
        rna_ecfp_cache[feat_set] = []
        for cdata in tqdm(complexes, desc=f"RNA hashes feat={feat_set}", leave=False):
            h = compute_rna_ecfp_hashes(
                cdata["rna_atoms"],
                cdata["rna_bond_graph"],
                MAX_RNA_DEPTH,
                feat_set
            )
            rna_ecfp_cache[feat_set].append(h)

    log.info("Pre-computing ligand ECFP hashes (max_depth={MAX_LIG_DEPTH})...")
    lig_ecfp_cache = []
    for cdata in tqdm(complexes, desc="Ligand hashes", leave=False):
        h = compute_ligand_ecfp_hashes(cdata["rdkit_mol"], MAX_LIG_DEPTH)
        lig_ecfp_cache.append(h)

    # 3. Full parameter sweep
    total_combos = (len(CUTOFFS) * len(RNA_DEPTHS) * len(LIG_DEPTHS) *
                    len(FP_SIZES) * len(FEAT_SETS))
    log.info(f"\nRunning {total_combos} FP combinations...")

    combo_idx = 0
    saved_files = []
    errors = []

    pbar = tqdm(total=total_combos, desc="FP grid")

    for feat_set in FEAT_SETS:
        for cutoff in CUTOFFS:
            cutoff_key = f"{cutoff:.1f}"
            for rna_depth in RNA_DEPTHS:
                for lig_depth in LIG_DEPTHS:
                    for fp_size in FP_SIZES:

                        out_name = (f"rlec_c{cutoff_key}_r{rna_depth}_"
                                    f"l{lig_depth}_s{fp_size}_f{feat_set}.npz")
                        out_path = FEAT_DIR / out_name

                        # Skip if already computed (resume support)
                        if out_path.exists():
                            pbar.update(1)
                            combo_idx += 1
                            continue

                        try:
                            X = np.zeros((n, fp_size), dtype=np.float32)

                            for ci, cdata in enumerate(complexes):
                                contacts = cdata["contacts"].get(cutoff, [])
                                if not contacts:
                                    # Zero vector — no contacts at this cutoff
                                    continue

                                bits = rlec_bits_for_complex(
                                    rna_ecfp_cache[feat_set][ci],
                                    lig_ecfp_cache[ci],
                                    contacts,
                                    rna_depth,
                                    lig_depth,
                                )
                                if bits:
                                    X[ci] = fold_bits(bits, fp_size)

                            np.savez_compressed(out_path,
                                                X=X, y=y, ids=pdb_ids)
                            saved_files.append(out_name)

                        except Exception as e:
                            errors.append({
                                "combo": out_name,
                                "error": str(e)
                            })
                            log.error(f"Error at {out_name}: {e}")

                        pbar.update(1)
                        combo_idx += 1

    pbar.close()

    # 4. Save run log
    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    run_log = {
        "timestamp"     : t_start.isoformat(),
        "elapsed_sec"   : elapsed,
        "n_complexes"   : n,
        "n_combos_total": total_combos,
        "n_saved"       : len(saved_files),
        "n_errors"      : len(errors),
        "errors"        : errors,
        "grid": {
            "cutoffs"   : CUTOFFS,
            "rna_depths": RNA_DEPTHS,
            "lig_depths": LIG_DEPTHS,
            "fp_sizes"  : FP_SIZES,
            "feat_sets" : FEAT_SETS,
        },
        "output_dir"    : str(FEAT_DIR),
    }
    log_path = LOG_DIR / "step04_fingerprint.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    log.info(f"\n{'=' * 60}")
    log.info(f"Done. Saved {len(saved_files)} FP matrices in {elapsed:.1f}s")
    log.info(f"Errors: {len(errors)}")
    log.info(f"Output: {FEAT_DIR}")
    log.info(f"Log:    {log_path}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
