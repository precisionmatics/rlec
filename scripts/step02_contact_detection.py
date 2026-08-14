"""
STEP 02 — RNA-Ligand Contact Detection
RLEC: RNA-Ligand Extended Connectivity Fingerprint

For each of the 143 complexes, computes RNA atom – ligand atom contact pairs
at ALL 7 distance cutoffs (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 Å) in one pass.

Approach:
  - RNA parsed via Biopython (NOT RDKit — known broken for PDB: GitHub #6501)
  - Ligand parsed via RDKit from SDF (reliable)
  - Contact detection via scipy cKDTree (fast, exact)

Outputs:
  - data/contacts/{pdb_id}_contacts.pkl  — dict keyed by cutoff
  - data/contacts/summary.csv            — per-complex contact counts per cutoff
  - logs/step02_contact_detection.json   — run log
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree

from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_CSV  = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")
CONTACT_DIR = ROOT / "data" / "contacts"
LOG_DIR   = ROOT / "logs"
CONTACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
CUTOFFS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

# STRICT WHITELIST: only these residue names count as genuine RNA/DNA nucleotide atoms.
# Standard nucleotides appear as ATOM records (het_flag=' ').
# Modified nucleotides appear as HETATM (het_flag='H_*') — include only curated list.
RNA_STANDARD = {"A", "U", "G", "C", "DA", "DT", "DG", "DC",
                "ADE", "URA", "GUA", "CYT", "THY"}

RNA_MODIFIED = {
    # Pseudouridine, methylated bases, iodo/bromo-substituted
    "PSU", "1MA", "5MC", "7MG", "OMC", "OMG", "2MG", "M2G",
    "OMU", "H2U", "PYI", "5BU", "4SU", "QUO", "YYG", "G7M",
    "70U", "2MA", "1MG", "3MU", "MIA", "A2M", "P5P", "5MU",
    "RSP", "RIA", "SSU", "T6A", "MRD", "PRN", "QBT", "RSQ",
    "CB2", "5IU", "YG", "A23", "23G", "50N", "50L",
    # Common Watson-Crick analogs
    "2AP", "DAP", "DI", "I", "INO", "XAN",
}

RNA_NUCLEOTIDES = RNA_STANDARD | RNA_MODIFIED

# Everything NOT in RNA_NUCLEOTIDES is excluded: protein residues,
# metal ions (MG, K, NA, ZN, BA, IRI, CS ...), water (HOH, WAT),
# crystallization agents (NCO, GOL, EDO, GLC, FRU, SIN, MPD, ACT ...),
# free nucleotides not part of chain (GTP, GDP, ATP, ADP, SPM, LCC, CBV ...).

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# ── RNA parsing ───────────────────────────────────────────────────────────────

def parse_rna_atoms(pdb_path: str) -> list[dict]:
    """
    Parse RNA PDB file with Biopython. Returns a list of atom records,
    one dict per heavy atom (no H).

    Each dict contains:
        idx          : int   — atom index in list
        name         : str   — atom name (e.g. 'N1', 'C4', 'O2*', 'P')
        resname      : str   — nucleotide residue name ('A','U','G','C', etc.)
        resnum       : int   — residue sequence number
        chain        : str   — chain ID
        coord        : np.ndarray shape (3,) — x,y,z in Å
        element      : str   — element symbol
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("rna", pdb_path)
    except Exception as e:
        return []

    atoms = []
    idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                het_flag = residue.get_id()[0]   # ' ' = ATOM, 'H_*' = HETATM, 'W' = water
                resname  = residue.get_resname().strip()
                resnum   = residue.get_id()[1]
                chain_id = chain.get_id()

                # STRICT WHITELIST: only genuine RNA/DNA nucleotide residues
                if het_flag == " ":
                    # ATOM record — accept only standard nucleotides
                    if resname not in RNA_STANDARD:
                        continue
                elif het_flag.startswith("H_"):
                    # HETATM record — accept only curated modified nucleotides
                    if resname not in RNA_MODIFIED:
                        continue
                else:
                    # Water ('W') or unknown — skip
                    continue

                for atom in residue.get_atoms():
                    element = atom.element.strip() if atom.element else ""
                    if element in ("H", "D", ""):   # skip hydrogens/unknown
                        continue
                    coord = atom.get_vector().get_array()
                    atoms.append({
                        "idx"    : idx,
                        "name"   : atom.get_name().strip(),
                        "resname": resname,
                        "resnum" : resnum,
                        "chain"  : chain_id,
                        "coord"  : coord,
                        "element": element,
                    })
                    idx += 1
    return atoms


# ── Ligand parsing ────────────────────────────────────────────────────────────

def parse_ligand_atoms(sdf_path: str) -> list[dict]:
    """
    Parse ligand SDF with RDKit. Returns a list of heavy atom records.

    Each dict contains:
        idx       : int
        atomic_num: int
        element   : str
        coord     : np.ndarray shape (3,)
        formal_charge : int
        is_aromatic   : bool
        n_heavy_nbrs  : int
        n_hs          : int (total H count)
        in_ring       : bool
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
    mol = None
    for m in suppl:
        if m is not None:
            mol = m
            break

    if mol is None:
        # try without sanitize
        suppl2 = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
        for m in suppl2:
            if m is not None:
                mol = m
                break

    if mol is None:
        return []

    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append({
            "idx"         : atom.GetIdx(),
            "atomic_num"  : atom.GetAtomicNum(),
            "element"     : atom.GetSymbol(),
            "coord"       : np.array([pos.x, pos.y, pos.z]),
            "formal_charge": atom.GetFormalCharge(),
            "is_aromatic" : atom.GetIsAromatic(),
            "n_heavy_nbrs": sum(1 for nb in atom.GetNeighbors()
                                if nb.GetAtomicNum() != 1),
            "n_hs"        : atom.GetTotalNumHs(),
            "in_ring"     : atom.IsInRing(),
        })
    return atoms


# ── Contact detection ─────────────────────────────────────────────────────────

def detect_contacts(rna_atoms: list[dict],
                    lig_atoms: list[dict],
                    cutoffs: list[float]) -> dict:
    """
    For each cutoff, find all (rna_idx, lig_idx, distance) pairs.
    Uses cKDTree on ligand coords, queries with RNA coords.

    Returns dict: { cutoff_float -> list of (rna_idx, lig_idx, dist) }
    """
    if not rna_atoms or not lig_atoms:
        return {c: [] for c in cutoffs}

    rna_coords = np.array([a["coord"] for a in rna_atoms])   # (N_rna, 3)
    lig_coords = np.array([a["coord"] for a in lig_atoms])   # (N_lig, 3)

    lig_tree = cKDTree(lig_coords)

    # query at max cutoff — then filter for smaller cutoffs (efficient)
    max_cutoff = max(cutoffs)
    # For each RNA atom, find all lig atoms within max_cutoff
    pairs_at_max = lig_tree.query_ball_point(rna_coords, r=max_cutoff)

    # Build full distance matrix only for pairs within max_cutoff
    contacts = {c: [] for c in cutoffs}

    for rna_idx, lig_indices in enumerate(pairs_at_max):
        if not lig_indices:
            continue
        rna_coord = rna_coords[rna_idx]
        for lig_idx in lig_indices:
            dist = float(np.linalg.norm(rna_coord - lig_coords[lig_idx]))
            for cutoff in cutoffs:
                if dist <= cutoff:
                    contacts[cutoff].append((rna_idx, lig_idx, dist))

    return contacts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("RLEC Step 02 — Contact Detection")
    log.info(f"Cutoffs: {CUTOFFS} Å")
    log.info("=" * 60)

    df = pd.read_csv(DATA_CSV)
    log.info(f"Loaded {len(df)} complexes from {DATA_CSV}")

    run_log = {
        "timestamp"   : datetime.now().isoformat(),
        "cutoffs"     : CUTOFFS,
        "n_complexes" : len(df),
        "results"     : {},
        "failed"      : [],
    }

    summary_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        pdb_id   = row["pdb"]
        rna_path = row["rna_pdb"]
        sdf_path = row["lig_sdf"]
        pkd      = row["pKd"]

        # ── parse ──
        rna_atoms = parse_rna_atoms(rna_path)
        lig_atoms = parse_ligand_atoms(sdf_path)

        if not rna_atoms:
            log.warning(f"  {pdb_id}: RNA parse failed — {rna_path}")
            run_log["failed"].append({"pdb": pdb_id, "reason": "rna_parse_failed"})
            continue
        if not lig_atoms:
            log.warning(f"  {pdb_id}: Ligand parse failed — {sdf_path}")
            run_log["failed"].append({"pdb": pdb_id, "reason": "lig_parse_failed"})
            continue

        # ── contacts at all cutoffs ──
        contacts = detect_contacts(rna_atoms, lig_atoms, CUTOFFS)

        # ── save ──
        out = {
            "pdb_id"   : pdb_id,
            "pKd"      : pkd,
            "rna_atoms": rna_atoms,
            "lig_atoms": lig_atoms,
            "contacts" : contacts,   # {cutoff: [(rna_idx, lig_idx, dist), ...]}
        }
        out_path = CONTACT_DIR / f"{pdb_id}_contacts.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(out, f, protocol=4)

        # ── summary row ──
        row_data = {"pdb": pdb_id, "pKd": pkd,
                    "n_rna_atoms": len(rna_atoms),
                    "n_lig_atoms": len(lig_atoms)}
        for c in CUTOFFS:
            row_data[f"n_contacts_{c}A"] = len(contacts[c])
        summary_rows.append(row_data)

        run_log["results"][pdb_id] = {
            "n_rna_atoms": len(rna_atoms),
            "n_lig_atoms": len(lig_atoms),
            "n_contacts" : {str(c): len(contacts[c]) for c in CUTOFFS},
        }

    # ── Summary CSV ──
    summary_df = pd.DataFrame(summary_rows)
    summary_path = CONTACT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ── Print stats ──
    log.info("\n" + "=" * 60)
    log.info("CONTACT STATISTICS PER CUTOFF")
    log.info("=" * 60)
    for c in CUTOFFS:
        col = f"n_contacts_{c}A"
        vals = summary_df[col]
        log.info(
            f"  {c:.1f} Å — mean: {vals.mean():.1f}  "
            f"median: {vals.median():.0f}  "
            f"min: {vals.min():.0f}  max: {vals.max():.0f}  "
            f"zero: {(vals == 0).sum()}"
        )

    log.info(f"\nSuccessful: {len(summary_rows)} / {len(df)}")
    log.info(f"Failed:     {len(run_log['failed'])}")

    # ── Save log ──
    log_path = LOG_DIR / "step02_contact_detection.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    log.info(f"\nLog saved: {log_path}")
    log.info(f"Summary:   {summary_path}")
    log.info(f"Contacts:  {CONTACT_DIR}/ ({len(summary_rows)} .pkl files)")

    return 0 if not run_log["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
