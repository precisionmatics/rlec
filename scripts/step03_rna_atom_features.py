"""
STEP 03 — RNA Atom Feature Engineering
RLEC: RNA-Ligand Extended Connectivity Fingerprint

For every RNA atom in every complex:
  1. Build covalent bond graph from atom coordinates (distance < 2.0 Å)
  2. Assign PBS group  : Phosphate=0 / Sugar=1 / Base=2
  3. Assign nucleotide type : A=0, U=1, G=2, C=3, T=4, modified=5
  4. Assign formal charge    : OP1/OP2 → -1, else 0
  5. Assign is_aromatic      : base ring atoms → True
  6. Assign in_ring          : base + ribose ring atoms → True
  7. Compute n_heavy_nbrs    : count bonded heavy atom neighbors from graph

Defines 4 feature sets used by all downstream RLEC FP computation:
  feat0 : basic ECFP    (atomic_num, formal_charge, n_heavy_nbrs, is_aromatic, in_ring)
  feat1 : feat0 + nucleotide type
  feat2 : feat0 + PBS group
  feat3 : feat0 + nucleotide type + PBS group  (all RNA features)

Updates all pkl files in data/contacts/ with enriched RNA atom dicts.
Saves validation stats to logs/step03_rna_features.json
"""

import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree
from collections import Counter
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CONTACT_DIR = ROOT / "data" / "contacts"
LOG_DIR     = ROOT / "logs"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── Element → atomic number ───────────────────────────────────────────────────
ELEM_TO_ANUM = {
    'H':1,  'C':6,  'N':7,  'O':8,  'F':9,
    'P':15, 'S':16, 'Cl':17,'Br':35,'I':53,
    'Mg':12,'K':19, 'Na':11,'Ca':20,'Zn':30,
    'Fe':26,'Mn':25,'Cu':29,'Co':27,'Ni':28,
}

# ── Bond distance cutoff (Å) ──────────────────────────────────────────────────
# Standard covalent bond lengths: C-C ~1.54, C-N ~1.47, C-O ~1.43, P-O ~1.60, P=O ~1.48
# Element-pair-specific bond distance cutoffs (Å).
# Tight limits for common pairs to avoid spurious 1-3 ring contacts;
# generous limit for C-I to cover 5-iodouridine (~2.14 Å).
_BOND_LIMITS = {
    frozenset(["C", "C"]) : 1.62,
    frozenset(["C", "N"]) : 1.55,
    frozenset(["C", "O"]) : 1.55,
    frozenset(["C", "S"]) : 1.90,
    frozenset(["C", "P"]) : 1.90,
    frozenset(["N", "N"]) : 1.50,
    frozenset(["N", "O"]) : 1.50,
    frozenset(["P", "O"]) : 1.72,
    frozenset(["C", "F"]) : 1.42,
    frozenset(["C", "Cl"]): 1.85,
    frozenset(["C", "Br"]): 2.05,
    frozenset(["C", "I"]) : 2.25,   # 5-iodouridine C5-I bond
    frozenset(["O", "P"]) : 1.72,
}
BOND_CUTOFF_DEFAULT = 2.0   # fallback for unlisted pairs

# ── PBS GROUP assignment by atom name ─────────────────────────────────────────
# Phosphate group atoms
PHOSPHATE_NAMES = {
    "P",
    "OP1", "OP2", "OP3",   # non-bridging phosphate oxygens (IUPAC names)
    "O1P", "O2P", "O3P",   # alternative naming convention
}

# Sugar (ribose) atoms — primed names, with both ' and * variants
SUGAR_NAMES = {
    "C1'", "C2'", "C3'", "C4'", "C5'",
    "O2'", "O4'", "O5'", "O3'",
    "C1*", "C2*", "C3*", "C4*", "C5*",
    "O2*", "O4*", "O5*", "O3*",
}
# O5' and O3' are bridging oxygens — classified as Sugar here (fingeRNAt convention)

# Base aromatic ring atoms (present in both purines and pyrimidines)
# Purines: N1 C2 N3 C4 C5 C6 N7 C8 N9 + exocyclic (N2 N6 O6 O2 N4 O4)
# Pyrimidines: N1 C2 N3 C4 C5 C6 + exocyclic
AROMATIC_BASE_RING = {
    "N1", "C2", "N3", "C4", "C5", "C6",   # pyrimidine ring (all nucleotides)
    "N7", "C8", "N9",                       # imidazole ring (purines only)
}

# All base ring atoms (in any ring, including non-aromatic parts)
BASE_RING_ATOMS = AROMATIC_BASE_RING  # all nucleobase ring atoms are aromatic

# Exocyclic base atoms (not in ring but still "base" group)
BASE_EXOCYCLIC = {
    "N2", "N4", "N6",    # amino groups
    "O2", "O4", "O6",    # carbonyl oxygens
    "C2M",               # methyl (modified)
}


def pbs_group(atom_name: str) -> int:
    """Assign PBS group: 0=Phosphate, 1=Sugar, 2=Base."""
    name = atom_name.strip()
    if name in PHOSPHATE_NAMES:
        return 0   # Phosphate
    if name in SUGAR_NAMES or "'" in name or "*" in name:
        return 1   # Sugar
    return 2       # Base


def nucleotide_type(resname: str) -> int:
    """Assign nucleotide type: A=0, U=1, G=2, C=3, T=4, modified=5."""
    r = resname.strip().upper()
    if r in {"A", "DA", "ADE"}:   return 0
    if r in {"U", "URA"}:         return 1
    if r in {"G", "DG", "GUA"}:   return 2
    if r in {"C", "DC", "CYT"}:   return 3
    if r in {"T", "DT", "THY"}:   return 4
    return 5   # modified nucleotide


def formal_charge(atom_name: str) -> int:
    """Assign formal charge. Phosphate non-bridging oxygens carry -1."""
    name = atom_name.strip()
    if name in {"OP1", "OP2", "O1P", "O2P"}:
        return -1
    return 0


def is_aromatic(atom_name: str, pbs: int) -> int:
    """1 if atom is in an aromatic nucleobase ring, else 0."""
    return int(pbs == 2 and atom_name.strip() in AROMATIC_BASE_RING)


def in_ring(atom_name: str, pbs: int) -> int:
    """1 if atom is in any ring (nucleobase ring or ribose O4' ring)."""
    name = atom_name.strip()
    if pbs == 2 and name in BASE_RING_ATOMS:
        return 1
    # Ribose ring: C1', C2', C3', C4', O4' form the 5-membered ring
    if name in {"C1'", "C2'", "C3'", "C4'", "O4'",
                "C1*", "C2*", "C3*", "C4*", "O4*"}:
        return 1
    return 0


# ── Build covalent bond graph ──────────────────────────────────────────────────

def _bond_cutoff(elem_a: str, elem_b: str) -> float:
    """Return the element-pair-specific max bond distance."""
    return _BOND_LIMITS.get(frozenset([elem_a, elem_b]), BOND_CUTOFF_DEFAULT)


def build_bond_graph(atoms: list[dict]) -> dict[int, list[int]]:
    """
    Build adjacency dict {atom_idx: [bonded_atom_idxs]} using element-pair-specific
    distance cutoffs. cKDTree with max cutoff for candidate pairs, then filter.
    """
    if len(atoms) < 2:
        return {a["idx"]: [] for a in atoms}

    coords = np.array([a["coord"] for a in atoms])
    elems  = [a["element"] for a in atoms]
    tree   = cKDTree(coords)

    # Query candidates within global max cutoff
    max_cutoff = max(_BOND_LIMITS.values())
    pairs = tree.query_pairs(r=max_cutoff, output_type="ndarray")

    graph = {a["idx"]: [] for a in atoms}
    for i, j in pairs:
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        limit = _bond_cutoff(elems[i], elems[j])
        if dist <= limit:
            ai = atoms[i]["idx"]
            aj = atoms[j]["idx"]
            graph[ai].append(aj)
            graph[aj].append(ai)

    return graph


# ── Enrich RNA atoms with all features ────────────────────────────────────────

def enrich_rna_atoms(atoms: list[dict], graph: dict) -> list[dict]:
    """
    Add feature fields to each RNA atom dict in-place and return enriched list.

    Added fields:
        pbs           : int  — 0=Phosphate, 1=Sugar, 2=Base
        nuc_type      : int  — 0=A, 1=U, 2=G, 3=C, 4=T, 5=modified
        formal_charge : int  — -1 for OP1/OP2, else 0
        is_aromatic   : int  — 1 if in aromatic nucleobase ring
        in_ring       : int  — 1 if in any ring
        n_heavy_nbrs  : int  — count of bonded heavy atom neighbors
        atomic_num    : int  — atomic number from element symbol
    """
    for atom in atoms:
        idx  = atom["idx"]
        name = atom["name"]
        res  = atom["resname"]
        elem = atom["element"]

        pbs_val   = pbs_group(name)
        nuc_val   = nucleotide_type(res)
        fc_val    = formal_charge(name)
        arom_val  = is_aromatic(name, pbs_val)
        ring_val  = in_ring(name, pbs_val)
        anum_val  = ELEM_TO_ANUM.get(elem, 0)

        nbrs = graph.get(idx, [])
        n_heavy = len(nbrs)   # graph already has only heavy atoms (no H in RNA parse)

        atom["pbs"]           = pbs_val
        atom["nuc_type"]      = nuc_val
        atom["formal_charge"] = fc_val
        atom["is_aromatic"]   = arom_val
        atom["in_ring"]       = ring_val
        atom["n_heavy_nbrs"]  = n_heavy
        atom["atomic_num"]    = anum_val

    return atoms


# ── Feature tuple builders (4 feature sets) ───────────────────────────────────

def atom_features(atom: dict, feature_set: int) -> tuple:
    """
    Return a feature tuple for hashing, given the feature set index.

    feat0: basic ECFP — (atomic_num, formal_charge, n_heavy_nbrs, is_aromatic, in_ring)
    feat1: feat0 + nucleotide type
    feat2: feat0 + PBS group
    feat3: feat0 + nucleotide type + PBS group  (all RNA features)
    """
    base = (
        atom["atomic_num"],
        atom["formal_charge"],
        atom["n_heavy_nbrs"],
        atom["is_aromatic"],
        atom["in_ring"],
    )
    if feature_set == 0:
        return base
    elif feature_set == 1:
        return base + (atom["nuc_type"],)
    elif feature_set == 2:
        return base + (atom["pbs"],)
    elif feature_set == 3:
        return base + (atom["nuc_type"], atom["pbs"])
    raise ValueError(f"Unknown feature_set: {feature_set}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("RLEC Step 03 — RNA Atom Feature Engineering")
    log.info("=" * 60)

    pkl_files = sorted(CONTACT_DIR.glob("*_contacts.pkl"))
    log.info(f"Found {len(pkl_files)} contact pkl files")

    run_log = {
        "timestamp"    : datetime.now().isoformat(),
        "n_complexes"  : len(pkl_files),
        "pbs_totals"   : Counter(),
        "nuc_type_totals": Counter(),
        "bond_degree"  : [],   # distribution of n_heavy_nbrs
        "results"      : {},
        "issues"       : [],
    }

    pbs_total   = Counter()
    nuc_total   = Counter()
    degree_all  = []

    for pkl_path in tqdm(pkl_files, desc="Enriching RNA atoms"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        pdb_id    = data["pdb_id"]
        rna_atoms = data["rna_atoms"]

        if not rna_atoms:
            run_log["issues"].append({"pdb": pdb_id, "issue": "no rna atoms"})
            continue

        # 1. Build covalent bond graph
        graph = build_bond_graph(rna_atoms)

        # 2. Enrich atoms with all features
        rna_atoms = enrich_rna_atoms(rna_atoms, graph)
        data["rna_atoms"]    = rna_atoms
        data["rna_bond_graph"] = graph   # save graph too

        # 3. Per-complex stats
        pbs_counts  = Counter(a["pbs"]      for a in rna_atoms)
        nuc_counts  = Counter(a["nuc_type"] for a in rna_atoms)
        degrees     = [a["n_heavy_nbrs"]    for a in rna_atoms]

        pbs_total  += pbs_counts
        nuc_total  += nuc_counts
        degree_all.extend(degrees)

        # Check: flag atoms with n_heavy_nbrs == 0 (isolated atoms — sign of parse issue)
        isolated = sum(1 for d in degrees if d == 0)
        if isolated > 0:
            run_log["issues"].append({
                "pdb": pdb_id,
                "issue": f"{isolated} isolated RNA atoms (no bonds found)"
            })

        run_log["results"][pdb_id] = {
            "n_rna_atoms" : len(rna_atoms),
            "pbs"         : dict(pbs_counts),
            "nuc_type"    : dict(nuc_counts),
            "isolated"    : isolated,
            "mean_degree" : float(np.mean(degrees)) if degrees else 0,
        }

        # 4. Save enriched pkl
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=4)

    # ── Summary ──
    pbs_labels  = {0: "Phosphate", 1: "Sugar", 2: "Base"}
    nuc_labels  = {0:"A", 1:"U", 2:"G", 3:"C", 4:"T", 5:"modified"}

    log.info("\n" + "=" * 60)
    log.info("PBS GROUP DISTRIBUTION (all atoms, all complexes)")
    log.info("=" * 60)
    total_atoms = sum(pbs_total.values())
    for k in sorted(pbs_total):
        pct = 100 * pbs_total[k] / total_atoms
        log.info(f"  {pbs_labels[k]:12s}: {pbs_total[k]:7d}  ({pct:.1f}%)")

    log.info("\nNUCLEOTIDE TYPE DISTRIBUTION")
    log.info("=" * 60)
    for k in sorted(nuc_total):
        pct = 100 * nuc_total[k] / total_atoms
        log.info(f"  {nuc_labels[k]:12s}: {nuc_total[k]:7d}  ({pct:.1f}%)")

    log.info("\nBOND DEGREE DISTRIBUTION (n_heavy_nbrs per RNA atom)")
    log.info("=" * 60)
    deg_arr = np.array(degree_all)
    for d in range(8):
        cnt = int((deg_arr == d).sum())
        pct = 100 * cnt / len(deg_arr)
        if cnt > 0:
            log.info(f"  degree {d}: {cnt:7d}  ({pct:.1f}%)")

    log.info(f"\n  mean={deg_arr.mean():.2f}  std={deg_arr.std():.2f}  "
             f"min={deg_arr.min()}  max={deg_arr.max()}")

    if run_log["issues"]:
        log.warning(f"\nIssues found ({len(run_log['issues'])}):")
        for iss in run_log["issues"]:
            log.warning(f"  {iss}")

    # ── Save log ──
    run_log["pbs_totals"]    = dict(pbs_total)
    run_log["nuc_type_totals"] = dict(nuc_total)
    run_log["bond_degree_mean"] = float(deg_arr.mean())
    run_log["bond_degree_max"]  = int(deg_arr.max())

    log_path = LOG_DIR / "step03_rna_features.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    log.info(f"\nLog saved: {log_path}")
    log.info("All pkl files updated with enriched RNA atom features + bond graph.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
