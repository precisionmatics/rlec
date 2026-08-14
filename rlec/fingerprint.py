"""
RLEC — RNA-Ligand Extended Connectivity Fingerprint

Adapts the PLEC fingerprint concept (Wójcikowski et al., Bioinformatics 2019)
to RNA-ligand systems.

Algorithm (per contact pair):
    rna_ecfp  = Morgan-style hashes for RNA atom  (depth 0..rna_depth)
    lig_ecfp  = Morgan-style hashes for ligand atom (depth 0..lig_depth)
    bits     += [hash(r,l) for r,l in zip_longest(rna_ecfp, lig_ecfp)]
Then fold all bits into a count vector of length fp_size.

Usage:
    from rlec import RLECFingerprint
    fp = RLECFingerprint(rna_depth=6, ligand_depth=3, fp_size=4096,
                         feat_set=1, cutoff=6.0)
    vec = fp.transform("path/to/rna.pdb", "path/to/ligand.sdf")
"""

from __future__ import annotations

import copy
import warnings
from itertools import zip_longest
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem

if TYPE_CHECKING:
    from rlec import __version__ as _version

warnings.filterwarnings("ignore")

# ── RNA residue whitelist ──────────────────────────────────────────────────────
# RDKit cannot reliably parse RNA PDB files (spurious/missing bonds, GitHub #6501).
# We use Biopython and filter strictly to genuine RNA/DNA nucleotide residues.

_RNA_STANDARD = {"A", "U", "G", "C", "DA", "DT", "DG", "DC",
                 "ADE", "URA", "GUA", "CYT", "THY"}

_RNA_MODIFIED = {
    "PSU", "1MA", "5MC", "7MG", "OMC", "OMG", "2MG", "M2G",
    "OMU", "H2U", "PYI", "5BU", "4SU", "QUO", "YYG", "G7M",
    "70U", "2MA", "1MG", "3MU", "MIA", "A2M", "P5P", "5MU",
    "RSP", "RIA", "SSU", "T6A", "MRD", "PRN", "QBT", "RSQ",
    "CB2", "5IU", "YG", "A23", "23G", "50N", "50L",
    "2AP", "DAP", "DI", "I", "INO", "XAN",
}

# ── RNA atom feature tables ────────────────────────────────────────────────────
_ELEM_TO_ANUM = {
    "H": 1,  "C": 6,  "N": 7,  "O": 8,  "F": 9,
    "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
    "Mg": 12, "K": 19, "Na": 11, "Ca": 20, "Zn": 30,
    "Fe": 26, "Mn": 25, "Cu": 29, "Co": 27, "Ni": 28,
}

# Element-pair-specific bond distance cutoffs (Å) — avoids spurious 1-3 ring bonds.
# C-I entry covers 5-iodouridine (~2.14 Å).
_BOND_LIMITS = {
    frozenset(["C", "C"]):  1.62,
    frozenset(["C", "N"]):  1.55,
    frozenset(["C", "O"]):  1.55,
    frozenset(["C", "S"]):  1.90,
    frozenset(["C", "P"]):  1.90,
    frozenset(["N", "N"]):  1.50,
    frozenset(["N", "O"]):  1.50,
    frozenset(["P", "O"]):  1.72,
    frozenset(["O", "P"]):  1.72,
    frozenset(["C", "F"]):  1.42,
    frozenset(["C", "Cl"]): 1.85,
    frozenset(["C", "Br"]): 2.05,
    frozenset(["C", "I"]):  2.25,
}
_BOND_CUTOFF_DEFAULT = 2.0

_PHOSPHATE_NAMES = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"}
_SUGAR_NAMES = {
    "C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O4'", "O5'", "O3'",
    "C1*", "C2*", "C3*", "C4*", "C5*", "O2*", "O4*", "O5*", "O3*",
}
_AROMATIC_BASE_RING = {"N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"}
_RIBOSE_RING = {"C1'", "C2'", "C3'", "C4'", "O4'", "C1*", "C2*", "C3*", "C4*", "O4*"}


# ── RNA parsing ────────────────────────────────────────────────────────────────

def _parse_rna_atoms(pdb_path: str) -> list[dict]:
    """
    Parse RNA PDB with Biopython. Returns list of heavy-atom dicts, filtered
    to genuine RNA/DNA nucleotide residues (strict whitelist — excludes protein,
    metal ions, water, crystallization agents).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rna", pdb_path)

    atoms: list[dict] = []
    idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                het_flag = residue.get_id()[0]
                resname = residue.get_resname().strip()

                if het_flag == " ":
                    if resname not in _RNA_STANDARD:
                        continue
                elif het_flag.startswith("H_"):
                    if resname not in _RNA_MODIFIED:
                        continue
                else:
                    continue

                resnum = residue.get_id()[1]
                chain_id = chain.get_id()

                for atom in residue.get_atoms():
                    element = atom.element.strip() if atom.element else ""
                    if element in ("H", "D", ""):
                        continue
                    atoms.append({
                        "idx":     idx,
                        "name":    atom.get_name().strip(),
                        "resname": resname,
                        "resnum":  resnum,
                        "chain":   chain_id,
                        "coord":   atom.get_vector().get_array(),
                        "element": element,
                    })
                    idx += 1
    return atoms


def _build_bond_graph(atoms: list[dict]) -> dict[int, list[int]]:
    """Build covalent bond adjacency dict using element-pair-specific distance cutoffs."""
    if len(atoms) < 2:
        return {a["idx"]: [] for a in atoms}

    coords = np.array([a["coord"] for a in atoms])
    elems = [a["element"] for a in atoms]
    tree = cKDTree(coords)

    max_cut = max(_BOND_LIMITS.values())
    pairs = tree.query_pairs(r=max_cut, output_type="ndarray")

    graph: dict[int, list[int]] = {a["idx"]: [] for a in atoms}
    for i, j in pairs:
        limit = _BOND_LIMITS.get(frozenset([elems[i], elems[j]]), _BOND_CUTOFF_DEFAULT)
        if np.linalg.norm(coords[i] - coords[j]) <= limit:
            ai, aj = atoms[i]["idx"], atoms[j]["idx"]
            graph[ai].append(aj)
            graph[aj].append(ai)
    return graph


def _enrich_rna_atoms(atoms: list[dict], graph: dict[int, list[int]]) -> list[dict]:
    """Add feature fields (pbs, nuc_type, formal_charge, is_aromatic, in_ring,
    n_heavy_nbrs, atomic_num) to each RNA atom dict in-place."""
    for atom in atoms:
        name = atom["name"]
        if name in _PHOSPHATE_NAMES:
            pbs = 0
        elif name in _SUGAR_NAMES or "'" in name or "*" in name:
            pbs = 1
        else:
            pbs = 2

        res = atom["resname"].strip().upper()
        if res in {"A", "DA", "ADE"}:   nuc = 0
        elif res in {"U", "URA"}:        nuc = 1
        elif res in {"G", "DG", "GUA"}:  nuc = 2
        elif res in {"C", "DC", "CYT"}:  nuc = 3
        elif res in {"T", "DT", "THY"}:  nuc = 4
        else:                             nuc = 5

        atom["pbs"]           = pbs
        atom["nuc_type"]      = nuc
        atom["formal_charge"] = -1 if name in {"OP1", "OP2", "O1P", "O2P"} else 0
        atom["is_aromatic"]   = int(pbs == 2 and name in _AROMATIC_BASE_RING)
        atom["in_ring"]       = int((pbs == 2 and name in _AROMATIC_BASE_RING)
                                    or name in _RIBOSE_RING)
        atom["n_heavy_nbrs"]  = len(graph.get(atom["idx"], []))
        atom["atomic_num"]    = _ELEM_TO_ANUM.get(atom["element"], 0)
    return atoms


# ── Ligand parsing ─────────────────────────────────────────────────────────────

def _mol_from_sdf(sdf_path: str):
    """Load the first valid RDKit mol from an SDF file."""
    for sanitize in (True, False):
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=sanitize)
        mol = next((m for m in suppl if m is not None), None)
        if mol is not None:
            return mol
    return None


def _lig_atoms_from_mol(mol) -> list[dict]:
    """Extract heavy-atom records from an RDKit mol with 3D conformer."""
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "Ligand Mol has no 3D conformer. "
            "Generate one with AllChem.EmbedMolecule() or load from an SDF with coordinates."
        )
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append({
            "idx":          atom.GetIdx(),
            "atomic_num":   atom.GetAtomicNum(),
            "element":      atom.GetSymbol(),
            "coord":        np.array([pos.x, pos.y, pos.z]),
            "formal_charge": atom.GetFormalCharge(),
            "is_aromatic":  atom.GetIsAromatic(),
            "n_heavy_nbrs": sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() != 1),
            "in_ring":      atom.IsInRing(),
        })
    return atoms


# ── Contact detection ──────────────────────────────────────────────────────────

def _detect_contacts(rna_atoms: list[dict], lig_atoms: list[dict],
                     cutoff: float) -> list[tuple]:
    """Return list of (rna_idx, lig_idx, dist) pairs within cutoff Å."""
    if not rna_atoms or not lig_atoms:
        return []
    rna_coords = np.array([a["coord"] for a in rna_atoms])
    lig_coords = np.array([a["coord"] for a in lig_atoms])
    tree = cKDTree(lig_coords)
    hits = tree.query_ball_point(rna_coords, r=cutoff)
    contacts = []
    for rna_i, lig_list in enumerate(hits):
        for lig_i in lig_list:
            dist = float(np.linalg.norm(rna_coords[rna_i] - lig_coords[lig_i]))
            contacts.append((rna_i, lig_i, dist))
    return contacts


# ── ECFP hash computation ──────────────────────────────────────────────────────

def _rna_atom_invariant(atom: dict, feat_set: int) -> int:
    """Depth-0 hash for an RNA atom under the given feature set (0–3)."""
    base = (atom["atomic_num"], atom["formal_charge"],
            atom["n_heavy_nbrs"], atom["is_aromatic"], atom["in_ring"])
    if feat_set == 1:
        features = base + (atom["nuc_type"],)
    elif feat_set == 2:
        features = base + (atom["pbs"],)
    elif feat_set == 3:
        features = base + (atom["nuc_type"], atom["pbs"])
    else:
        features = base
    return hash(features) & 0xFFFFFFFF


def _compute_rna_ecfp_hashes(atoms: list[dict], graph: dict[int, list[int]],
                               max_depth: int, feat_set: int) -> dict[int, list[int]]:
    """Morgan-style hashes for all RNA atoms up to max_depth.
    Returns {atom_idx: [hash_d0, hash_d1, ..., hash_d_max]}.
    """
    current = {a["idx"]: _rna_atom_invariant(a, feat_set) for a in atoms}
    all_hashes = {a["idx"]: [current[a["idx"]]] for a in atoms}

    for depth in range(1, max_depth + 1):
        new_hashes = {}
        for atom in atoms:
            idx = atom["idx"]
            nbr_hashes = tuple(sorted(current[j] for j in graph.get(idx, [])))
            new_hashes[idx] = hash((depth, current[idx]) + nbr_hashes) & 0xFFFFFFFF
        current = new_hashes
        for idx, h in current.items():
            all_hashes[idx].append(h)

    return all_hashes


def _compute_lig_ecfp_hashes(mol, max_depth: int) -> dict[int, list[int]]:
    """Per-atom Morgan hashes at depths 0..max_depth via RDKit bitInfo.
    Returns {atom_idx: [hash_d0, ..., hash_d_max]}.
    """
    heavy_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    result: dict[int, list[int]] = {i: [] for i in heavy_idxs}

    for depth in range(max_depth + 1):
        bi: dict = {}
        AllChem.GetMorganFingerprint(mol, depth, bitInfo=bi)
        atom_to_hash: dict[int, int] = {}
        for bit_hash, atom_rad_list in bi.items():
            for atom_idx, radius in atom_rad_list:
                if radius == depth:
                    atom_to_hash[atom_idx] = bit_hash
        for idx in heavy_idxs:
            h = atom_to_hash.get(idx, result[idx][0] if result[idx] else 0)
            result[idx].append(h)

    return result


# ── Core RLEC bit generation ───────────────────────────────────────────────────

def _rlec_bits(rna_hashes: dict[int, list[int]], lig_hashes: dict[int, list[int]],
               contacts: list[tuple], rna_depth: int, lig_depth: int) -> list[int]:
    """
    Core RLEC algorithm (PLEC-style for RNA):
      For each contact (rna_i, lig_i):
        zip_longest(rna_ecfp[:rna_depth+1], lig_ecfp[:lig_depth+1]) → hash each pair
    """
    bits = []
    for rna_i, lig_i, _ in contacts:
        rna_env = rna_hashes.get(rna_i)
        lig_env = lig_hashes.get(lig_i)
        if rna_env is None or lig_env is None:
            continue
        rna_env = rna_env[:rna_depth + 1]
        lig_env = lig_env[:lig_depth + 1]
        rna_fill = rna_env[-1] if rna_env else 0
        lig_fill = lig_env[-1] if lig_env else 0
        for r_h, l_h in zip_longest(rna_env, lig_env, fillvalue=None):
            r_h = r_h if r_h is not None else rna_fill
            l_h = l_h if l_h is not None else lig_fill
            bits.append(hash((r_h, l_h)) & 0xFFFFFFFF)
    return bits


def _fold_bits(bits: list[int], fp_size: int) -> np.ndarray:
    """Fold 32-bit hashes into a count vector of length fp_size."""
    vec = np.zeros(fp_size, dtype=np.float32)
    for b in bits:
        vec[b % fp_size] += 1
    return vec


# ── Public API ─────────────────────────────────────────────────────────────────

class RLECFingerprint:
    """RNA-Ligand Extended Connectivity Fingerprint.

    Best-validated parameters (LOOCV r=0.7097 on 143 RNA-ligand complexes):
        rna_depth=6, ligand_depth=3, fp_size=4096, feat_set=1, cutoff=6.0

    Parameters
    ----------
    rna_depth : int
        Morgan iteration depth for RNA atoms (default 6).
    ligand_depth : int
        Morgan iteration depth for ligand atoms (default 3).
    fp_size : int
        Folded fingerprint size (default 4096).
    feat_set : int
        RNA atom feature set: 0=basic ECFP, 1=+nucleotide_type (best),
        2=+PBS_group, 3=+both (default 1).
    cutoff : float
        RNA-ligand contact distance cutoff in Angstroms (default 6.0).
    """

    def __init__(
        self,
        rna_depth: int = 6,
        ligand_depth: int = 3,
        fp_size: int = 4096,
        feat_set: int = 1,
        cutoff: float = 6.0,
    ):
        if feat_set not in (0, 1, 2, 3):
            raise ValueError(f"feat_set must be 0, 1, 2, or 3; got {feat_set}")
        if rna_depth < 0:
            raise ValueError(f"rna_depth must be >= 0; got {rna_depth}")
        if ligand_depth < 0:
            raise ValueError(f"ligand_depth must be >= 0; got {ligand_depth}")
        if fp_size < 1:
            raise ValueError(f"fp_size must be >= 1; got {fp_size}")
        if cutoff <= 0:
            raise ValueError(f"cutoff must be > 0; got {cutoff}")
        self.rna_depth = rna_depth
        self.ligand_depth = ligand_depth
        self.fp_size = fp_size
        self.feat_set = feat_set
        self.cutoff = cutoff

    def __repr__(self) -> str:
        return (
            f"RLECFingerprint(rna_depth={self.rna_depth}, "
            f"ligand_depth={self.ligand_depth}, "
            f"fp_size={self.fp_size}, "
            f"feat_set={self.feat_set}, "
            f"cutoff={self.cutoff})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RLECFingerprint):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def copy(self) -> "RLECFingerprint":
        """Return an independent copy of this fingerprint object."""
        return copy.copy(self)

    def get_params(self, deep: bool = True) -> dict:
        """Return parameters as a dict (sklearn-compatible)."""
        return {
            "rna_depth":    self.rna_depth,
            "ligand_depth": self.ligand_depth,
            "fp_size":      self.fp_size,
            "feat_set":     self.feat_set,
            "cutoff":       self.cutoff,
        }

    def set_params(self, **params) -> "RLECFingerprint":
        """Set parameters in-place and return self (sklearn-compatible)."""
        valid = set(self.get_params())
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"Unknown parameter: {k!r}. Valid: {sorted(valid)}")
            setattr(self, k, v)
        RLECFingerprint(**self.get_params())  # re-validate
        return self

    def to_dict(self) -> dict:
        """Serialise parameters to a plain dict (round-trips via from_dict)."""
        from rlec import __version__
        return {"rna_depth": self.rna_depth, "ligand_depth": self.ligand_depth,
                "fp_size": self.fp_size, "feat_set": self.feat_set,
                "cutoff": self.cutoff, "__version__": __version__}

    @classmethod
    def from_dict(cls, d: dict) -> "RLECFingerprint":
        """Reconstruct an instance from a dict produced by to_dict()."""
        params = {k: v for k, v in d.items() if not k.startswith("__")}
        required = {"rna_depth", "ligand_depth", "fp_size", "feat_set", "cutoff"}
        missing = required - set(params)
        if missing:
            raise KeyError(f"Missing required keys in dict: {missing}")
        return cls(**params)

    def transform(self, rna_pdb: str, ligand) -> np.ndarray:
        """Compute RLEC fingerprint for one RNA-ligand complex.

        Parameters
        ----------
        rna_pdb : str
            Path to RNA PDB file (parsed with Biopython).
        ligand : str or rdkit.Chem.Mol
            Path to ligand SDF file, or an RDKit Mol object with a 3D conformer.

        Returns
        -------
        np.ndarray, shape (fp_size,), dtype float32
            Count fingerprint vector. All-zero if no contacts found.
        """
        rna_atoms = _parse_rna_atoms(str(rna_pdb))
        if not rna_atoms:
            return np.zeros(self.fp_size, dtype=np.float32)

        rna_graph = _build_bond_graph(rna_atoms)
        rna_atoms = _enrich_rna_atoms(rna_atoms, rna_graph)

        mol = _mol_from_sdf(ligand) if isinstance(ligand, str) else ligand
        if mol is None:
            return np.zeros(self.fp_size, dtype=np.float32)

        lig_atoms = _lig_atoms_from_mol(mol)
        if not lig_atoms:
            return np.zeros(self.fp_size, dtype=np.float32)

        contacts = _detect_contacts(rna_atoms, lig_atoms, self.cutoff)
        if not contacts:
            return np.zeros(self.fp_size, dtype=np.float32)

        rna_hashes = _compute_rna_ecfp_hashes(rna_atoms, rna_graph,
                                               self.rna_depth, self.feat_set)
        lig_hashes = _compute_lig_ecfp_hashes(mol, self.ligand_depth)

        bits = _rlec_bits(rna_hashes, lig_hashes, contacts,
                          self.rna_depth, self.ligand_depth)
        return _fold_bits(bits, self.fp_size)

    def transform_batch(
        self,
        complexes: list[tuple],
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute RLEC fingerprints for a list of (rna_pdb, ligand) pairs.

        Parameters
        ----------
        complexes : list of (rna_pdb, ligand) tuples
            Each tuple as accepted by transform().
        n_jobs : int
            Number of parallel workers. 1 = serial, -1 = all CPUs (default 1).
        show_progress : bool
            Show tqdm progress bar (default True).

        Returns
        -------
        np.ndarray, shape (n, fp_size), dtype float32
        """
        n = len(complexes)
        if n == 0:
            return np.zeros((0, self.fp_size), dtype=np.float32)

        if n_jobs == 1:
            X = np.zeros((n, self.fp_size), dtype=np.float32)
            for i, (rna_pdb, ligand) in enumerate(
                tqdm(complexes, desc="RLEC", disable=not show_progress)
            ):
                X[i] = self.transform(rna_pdb, ligand)
            return X

        results = list(tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(self.transform)(rna, lig) for rna, lig in complexes
            ),
            total=n, desc="RLEC", disable=not show_progress,
        ))
        return np.vstack(results).astype(np.float32)

    def to_sparse(
        self,
        complexes: list[tuple],
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> sp.csr_matrix:
        """Compute fingerprints and return a scipy CSR sparse matrix.

        Useful when fp_size is large (≥ 32768) and most bits are zero.

        Parameters
        ----------
        complexes : list of (rna_pdb, ligand) tuples
        n_jobs : int
            Parallel workers (default 1).
        show_progress : bool
            Show tqdm bar (default True).

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n, fp_size), dtype float32
        """
        X = self.transform_batch(complexes, n_jobs=n_jobs, show_progress=show_progress)
        return sp.csr_matrix(X)

    def to_dataframe(
        self,
        complexes: list[tuple],
        ids: list[str] | None = None,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute fingerprints and return a pandas DataFrame.

        Parameters
        ----------
        complexes : list of (rna_pdb, ligand) tuples
        ids : list of str, optional
            Row identifiers (e.g. PDB IDs). If given, added as first column "id".
        n_jobs : int
            Parallel workers (default 1).
        show_progress : bool
            Show tqdm bar (default True).

        Returns
        -------
        pandas.DataFrame with columns bit_0 … bit_{fp_size-1} (and "id" if ids given)
        """
        if ids is not None and len(ids) != len(complexes):
            raise ValueError(
                f"len(ids)={len(ids)} does not match len(complexes)={len(complexes)}"
            )
        X = self.transform_batch(complexes, n_jobs=n_jobs, show_progress=show_progress)
        cols = [f"bit_{i}" for i in range(self.fp_size)]
        df = pd.DataFrame(X, columns=cols)
        if ids is not None:
            df.insert(0, "id", ids)
        return df
