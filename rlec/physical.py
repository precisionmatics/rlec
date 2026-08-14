"""
RLEC Physical Interaction Features
===================================
Computes 29 physics-based descriptors from an RNA-ligand 3D complex.

These features complement the RLEC fingerprint by capturing explicit
interaction energetics that bit-hashing alone cannot encode:

  Electrostatic  : Gasteiger-charge-weighted Coulomb sum at interface
  H-bond         : donor-acceptor pair count (r < 3.5 Å)
  Hydrophobic    : C/S···C/S contact count (r < 4.5 Å)
  Ionic          : RNA phosphate···cationic ligand atom count
  Aromatic       : π-stacking ring-atom pairs (r < 5.5 Å)
  Geometric      : contact count, density, distance statistics
  Ligand-global  : Gasteiger charge stats, RDKit 2D descriptors

Usage
-----
    from rlec.physical import compute_physical_features, PHYSICAL_FEATURE_NAMES

    feat = compute_physical_features("rna.pdb", "ligand.sdf")
    # feat: np.ndarray shape (29,), or None on parse failure

    # Combine with RLEC fingerprint
    from rlec import RLECFingerprint
    fp  = RLECFingerprint()
    vec = fp.transform("rna.pdb", "ligand.sdf")          # (4096,)
    phy = compute_physical_features("rna.pdb", "ligand.sdf")  # (29,)
    combined = np.concatenate([vec, phy])                 # (4125,)

Reference (electrostatic insight)
----------------------------------
Xia et al. 2025 (RLASIF, Comput. Biol. Chem.) showed that atomic charge
is the single most predictive surface feature for RNA-ligand binding affinity.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

warnings.filterwarnings("ignore")

# ── RNA residue whitelist (same as fingerprint.py) ────────────────────────────
_RNA_RESIDUES = {
    "A", "U", "G", "C", "T", "DA", "DT", "DG", "DC",
    "ADE", "URA", "GUA", "CYT", "THY",
    "PSU", "1MA", "5MC", "7MG", "OMC", "OMG", "2MG", "M2G",
    "OMU", "H2U", "4SU", "5MU", "5BU", "MIA", "INO", "2AP",
}

# Approximate partial charges for RNA heavy atoms (Pauling scale, simplified)
_RNA_PARTIAL_Q = {"N": -0.3, "O": -0.4, "P": +1.2, "C": 0.0, "S": -0.2}

# Interaction geometry thresholds
_HBOND_DIST   = 3.5   # Å  N/O donor → N/O acceptor
_HYDRO_DIST   = 4.5   # Å  C/S ··· C/S
_IONIC_DIST   = 5.0   # Å  RNA-P ··· cationic lig atom
_AROM_DIST    = 5.5   # Å  aromatic ··· aromatic
_CONTACT_DIST = 6.0   # Å  general contact cutoff

_HB_ELEMENTS = {"N", "O"}
_HYDRO_ELEMS = {"C", "S", "F", "Cl", "Br", "I"}

# ── Feature names (29 total) ──────────────────────────────────────────────────

PHYSICAL_FEATURE_NAMES: list[str] = [
    # Electrostatic (5)
    "E_elec_all",          # Σ q_rna × q_lig / r  over all contacts
    "E_elec_close",        # same, restricted to r < 4.0 Å
    "mean_q_lig_contact",  # mean Gasteiger charge of ligand atoms in contact
    "std_q_lig_contact",   # std  Gasteiger charge of ligand atoms in contact
    "mean_q_rna_contact",  # mean partial charge of RNA atoms in contact
    # H-bond (2)
    "n_hbond",             # H-bond donor/acceptor pair count
    "frac_hbond",          # n_hbond / n_contacts
    # Hydrophobic (2)
    "n_hydro",             # hydrophobic contact count
    "frac_hydro",          # n_hydro / n_contacts
    # Ionic / Aromatic (2)
    "n_ionic",             # RNA-P ··· cationic lig atom
    "n_arom_pi",           # aromatic ··· aromatic (π-stacking)
    # Geometric (6)
    "n_contacts",          # total contact pairs
    "n_rna_atoms_contact", # unique RNA atoms involved in contacts
    "contact_density",     # n_contacts / n_rna_atoms
    "mean_dist",           # mean contact distance
    "min_dist",            # minimum contact distance
    "std_dist",            # std of contact distances
    # Ligand Gasteiger charge stats (4)
    "lig_q_sum",           # sum of all Gasteiger charges
    "lig_q_std",           # std of all Gasteiger charges
    "lig_q_max",           # max Gasteiger charge (most positive atom)
    "lig_q_min",           # min Gasteiger charge (most negative atom)
    # Ligand RDKit 2D descriptors (8)
    "lig_n_atoms",
    "lig_mw",
    "lig_tpsa",
    "lig_hbd",
    "lig_hba",
    "lig_rot_bonds",
    "lig_rings",
    "lig_arom_rings",
]

assert len(PHYSICAL_FEATURE_NAMES) == 29


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_rna(pdb_path: str) -> list[dict]:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("rna", pdb_path)
    atoms = []
    for model in struct:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() not in _RNA_RESIDUES:
                    continue
                for atom in residue:
                    elem = (atom.element or atom.name[0]).strip()
                    atoms.append({
                        "coord": np.array(atom.coord, dtype=np.float32),
                        "elem":  elem,
                        "q":     _RNA_PARTIAL_Q.get(elem, 0.0),
                    })
    return atoms


def _parse_ligand(sdf_path: str):
    """Return (list[dict], Mol) or (None, None) on failure."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        for m in supplier:
            if m is not None:
                try:
                    Chem.SanitizeMol(m)
                    mol = m
                    break
                except Exception:
                    pass
    if mol is None or mol.GetNumConformers() == 0:
        return None, None

    AllChem.ComputeGasteigerCharges(mol)
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        gc  = float(atom.GetDoubleProp("_GasteigerCharge") or 0.0)
        if not np.isfinite(gc):
            gc = 0.0
        atoms.append({
            "coord":    np.array([pos.x, pos.y, pos.z], dtype=np.float32),
            "elem":     atom.GetSymbol(),
            "q":        gc,
            "fcharge":  atom.GetFormalCharge(),
            "aromatic": atom.GetIsAromatic(),
        })
    return atoms, mol


# ── Public API ────────────────────────────────────────────────────────────────

def compute_physical_features(
    rna_pdb:  str,
    lig_sdf:  str,
    cutoff:   float = 6.0,
) -> Optional[np.ndarray]:
    """Compute 29 physical interaction features for an RNA-ligand complex.

    Parameters
    ----------
    rna_pdb : str
        Path to RNA PDB file.
    lig_sdf : str
        Path to ligand SDF file (must contain 3D coordinates).
    cutoff : float
        Contact distance cutoff in Å (default 6.0).

    Returns
    -------
    np.ndarray of shape (29,) and dtype float32, or None if parsing fails.

    Feature names are available as ``rlec.physical.PHYSICAL_FEATURE_NAMES``.
    """
    rna_atoms = _parse_rna(rna_pdb)
    lig_atoms, mol = _parse_ligand(lig_sdf)
    if not rna_atoms or lig_atoms is None:
        return None

    rna_coords = np.array([a["coord"] for a in rna_atoms])
    lig_coords = np.array([a["coord"] for a in lig_atoms])

    # ── Build contact list ─────────────────────────────────────────────────────
    tree = cKDTree(lig_coords)
    contacts: list[tuple[int, int, float]] = []
    for ri, ra in enumerate(rna_atoms):
        for li in tree.query_ball_point(ra["coord"], cutoff):
            dist = float(np.linalg.norm(ra["coord"] - lig_atoms[li]["coord"]))
            contacts.append((ri, li, dist))

    if not contacts:
        return None

    dists = np.array([c[2] for c in contacts], dtype=np.float64)
    n     = len(contacts)

    # ── Electrostatic ──────────────────────────────────────────────────────────
    E_all   = sum(rna_atoms[ri]["q"] * lig_atoms[li]["q"] / max(d, 0.5)
                  for ri, li, d in contacts)
    E_close = sum(rna_atoms[ri]["q"] * lig_atoms[li]["q"] / max(d, 0.5)
                  for ri, li, d in contacts if d < 4.0)

    contact_lig_q = [lig_atoms[li]["q"] for _, li, _ in contacts]
    contact_rna_q = [rna_atoms[ri]["q"] for ri, _, _ in contacts]

    # ── H-bonds ───────────────────────────────────────────────────────────────
    n_hbond = sum(
        1 for ri, li, d in contacts
        if d < _HBOND_DIST
        and (
            (rna_atoms[ri]["elem"] in _HB_ELEMENTS and lig_atoms[li]["elem"] in _HB_ELEMENTS)
        )
    )

    # ── Hydrophobic ───────────────────────────────────────────────────────────
    n_hydro = sum(
        1 for ri, li, d in contacts
        if d < _HYDRO_DIST
        and rna_atoms[ri]["elem"] in _HYDRO_ELEMS
        and lig_atoms[li]["elem"] in _HYDRO_ELEMS
    )

    # ── Ionic ─────────────────────────────────────────────────────────────────
    n_ionic = sum(
        1 for ri, li, d in contacts
        if d < _IONIC_DIST
        and rna_atoms[ri]["elem"] == "P"
        and lig_atoms[li]["fcharge"] > 0
    )

    # ── Aromatic π-stacking ───────────────────────────────────────────────────
    n_arom = sum(
        1 for ri, li, d in contacts
        if d < _AROM_DIST
        and rna_atoms[ri]["elem"] in {"C", "N"}
        and lig_atoms[li]["aromatic"]
    )

    # ── Geometric ─────────────────────────────────────────────────────────────
    rna_contact_atoms = len({ri for ri, _, _ in contacts})
    n_rna             = len(rna_atoms)

    # ── Ligand global Gasteiger ───────────────────────────────────────────────
    all_lig_q = [a["q"] for a in lig_atoms]

    # ── Ligand RDKit descriptors ──────────────────────────────────────────────
    feat = np.array([
        # Electrostatic (5)
        E_all,
        E_close,
        float(np.mean(contact_lig_q)),
        float(np.std(contact_lig_q)) if n > 1 else 0.0,
        float(np.mean(contact_rna_q)),
        # H-bond (2)
        float(n_hbond),
        float(n_hbond / n),
        # Hydrophobic (2)
        float(n_hydro),
        float(n_hydro / n),
        # Ionic / Aromatic (2)
        float(n_ionic),
        float(n_arom),
        # Geometric (6)
        float(n),
        float(rna_contact_atoms),
        float(n / max(n_rna, 1)),
        float(np.mean(dists)),
        float(np.min(dists)),
        float(np.std(dists)),
        # Ligand Gasteiger (4)
        float(np.sum(all_lig_q)),
        float(np.std(all_lig_q)),
        float(np.max(all_lig_q)),
        float(np.min(all_lig_q)),
        # Ligand RDKit 2D (8)
        float(mol.GetNumHeavyAtoms()),
        float(Descriptors.MolWt(mol)),
        float(Descriptors.TPSA(mol)),
        float(Descriptors.NumHDonors(mol)),
        float(Descriptors.NumHAcceptors(mol)),
        float(Descriptors.NumRotatableBonds(mol)),
        float(Descriptors.RingCount(mol)),
        float(Descriptors.NumAromaticRings(mol)),
    ], dtype=np.float32)

    if not np.all(np.isfinite(feat)):
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    return feat


def compute_physical_batch(
    complexes: list[tuple[str, str]],
    cutoff:    float = 6.0,
    n_jobs:    int   = 1,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Compute physical features for a batch of complexes.

    Parameters
    ----------
    complexes : list of (rna_pdb, lig_sdf) tuples
    cutoff    : contact distance cutoff in Å
    n_jobs    : parallel workers (-1 = all CPUs)
    fill_value: value to use for failed complexes (default 0.0)

    Returns
    -------
    np.ndarray of shape (N, 29)
    """
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_physical_features)(rna, lig, cutoff)
        for rna, lig in complexes
    )

    n_feat = len(PHYSICAL_FEATURE_NAMES)
    out = np.full((len(complexes), n_feat), fill_value, dtype=np.float32)
    for i, feat in enumerate(results):
        if feat is not None:
            out[i] = feat
    return out
