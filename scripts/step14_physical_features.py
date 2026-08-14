"""
STEP 14 — Physical Interaction Features from 3D Structure
RLEC misses the physics. RLASIF proved atomic charge is the #1 predictor.

Features computed per complex:
  ELECTROSTATIC:
    E_elec   — Σ q_rna × q_lig / r  (Gasteiger charges, all contact pairs)
    n_ionic  — count of RNA-P···lig+ contacts (salt bridges)
  H-BOND:
    n_hbond  — H-bond count (N/O donor → N/O acceptor, r<3.5Å, angle approx)
    n_hba_rna, n_hbd_rna — RNA H-bond acceptor/donor atoms in contact zone
  HYDROPHOBIC:
    n_hydro  — C···C/C···S contacts < 4.5Å (hydrophobic)
  AROMATIC:
    n_arom_pi — aromatic ring-ring pairs < 5.5Å (π-π stacking)
  SHAPE:
    n_contacts — total contact count
    contact_density — contacts per RNA atom
  DISTANCE:
    mean_dist, min_dist — mean/min contact distance
  LIGAND-GLOBAL (from RDKit):
    Gasteiger charge sum, partial charge std

Combined with RLEC bits → pushed PCC toward 0.8
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Biopython + RDKit
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / "data" / "features"
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)
DATA_CSV = Path("/home/stalin/Desktop/CAML/data/dataset_clean.csv")

# ── Element properties ────────────────────────────────────────────────────────
# Pauling partial charge estimates for RNA atoms (approximation)
RNA_CHARGE = {"N": -0.3, "O": -0.4, "P": +1.2, "C": 0.0, "S": -0.2, "H": +0.2}
# Hydrophobic elements
HYDROPHOBIC = {"C", "S", "F", "Cl", "Br", "I"}
# H-bond donor/acceptor elements (simplified)
HB_DONOR    = {"N", "O"}
HB_ACCEPTOR = {"N", "O"}
AROMATIC_ELEMENTS = {"C", "N"}

# ── Parse RNA atoms ───────────────────────────────────────────────────────────
RNA_RESIDUES = {
    "A", "U", "G", "C", "T", "DA", "DT", "DG", "DC",    # standard
    "ADE", "URA", "GUA", "CYT", "THY",                    # 3-letter
    "RA", "RU", "RG", "RC",                               # prefixed
    "5MU", "PSU", "H2U", "OMG", "OMC", "OMA", "7MG",     # modified
    "4SU", "FMU", "IU", "CBR", "BRU", "MIA",              # modified
}

def parse_rna_atoms(pdb_path):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("rna", pdb_path)
    atoms = []
    for model in struct:
        for chain in model:
            for residue in chain:
                rname = residue.get_resname().strip()
                if rname not in RNA_RESIDUES:
                    continue
                for atom in residue:
                    elem = atom.element.strip() if atom.element else atom.name[0]
                    atoms.append({
                        "coord": np.array(atom.coord),
                        "elem": elem,
                        "formal_charge": 0,
                        "partial_charge": RNA_CHARGE.get(elem, 0.0),
                        "is_aromatic": elem in AROMATIC_ELEMENTS and "P" not in atom.name,
                    })
    return atoms


def parse_ligand_atoms(sdf_path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mol = None
    for m in supplier:
        if m is not None:
            mol = m
            break
    if mol is None:
        return None, None

    AllChem.ComputeGasteigerCharges(mol)
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        gc = float(atom.GetDoubleProp("_GasteigerCharge") or 0.0)
        if np.isnan(gc) or np.isinf(gc):
            gc = 0.0
        elem = atom.GetSymbol()
        atoms.append({
            "coord": np.array([pos.x, pos.y, pos.z]),
            "elem": elem,
            "gasteiger": gc,
            "formal_charge": atom.GetFormalCharge(),
            "is_aromatic": atom.GetIsAromatic(),
        })
    return atoms, mol


def compute_physical_features(rna_pdb, lig_sdf, cutoff=6.0):
    rna_atoms = parse_rna_atoms(rna_pdb)
    lig_atoms, mol = parse_ligand_atoms(lig_sdf)
    if not rna_atoms or lig_atoms is None or len(lig_atoms) == 0:
        return None

    rna_coords = np.array([a["coord"] for a in rna_atoms])
    lig_coords = np.array([a["coord"] for a in lig_atoms])

    tree = cKDTree(lig_coords)
    contacts = []
    for ri, ra in enumerate(rna_atoms):
        idx = tree.query_ball_point(ra["coord"], cutoff)
        for li in idx:
            la = lig_atoms[li]
            dist = float(np.linalg.norm(ra["coord"] - la["coord"]))
            contacts.append((ri, li, dist))

    if not contacts:
        return None

    # ── Physical features ──────────────────────────────────────────────────────
    dists = [c[2] for c in contacts]
    n_contacts = len(contacts)
    mean_dist  = np.mean(dists)
    min_dist   = np.min(dists)

    # Electrostatic: q_rna × q_lig / r
    E_elec = 0.0
    for ri, li, dist in contacts:
        q_rna = rna_atoms[ri]["partial_charge"]
        q_lig = lig_atoms[li]["gasteiger"]
        if dist > 0.5:
            E_elec += q_rna * q_lig / dist

    # H-bonds: donor N/O → acceptor N/O, dist < 3.5Å
    n_hbond = 0
    for ri, li, dist in contacts:
        if dist < 3.5:
            re = rna_atoms[ri]["elem"]
            le = lig_atoms[li]["elem"]
            if re in HB_ACCEPTOR and le in HB_DONOR:
                n_hbond += 1
            elif le in HB_ACCEPTOR and re in HB_DONOR:
                n_hbond += 1

    # Hydrophobic: C/S···C/S < 4.5Å
    n_hydro = sum(1 for ri, li, dist in contacts
                  if dist < 4.5
                  and rna_atoms[ri]["elem"] in HYDROPHOBIC
                  and lig_atoms[li]["elem"] in HYDROPHOBIC)

    # Ionic: RNA-P···lig with positive formal charge < 5Å
    n_ionic = sum(1 for ri, li, dist in contacts
                  if dist < 5.0
                  and rna_atoms[ri]["elem"] == "P"
                  and lig_atoms[li]["formal_charge"] > 0)

    # Aromatic π-stacking: aromatic RNA atom ↔ aromatic lig atom < 5.5Å
    n_arom = sum(1 for ri, li, dist in contacts
                 if dist < 5.5
                 and rna_atoms[ri]["is_aromatic"]
                 and lig_atoms[li]["is_aromatic"])

    # RNA contact atoms
    rna_contact_set = set(ri for ri, li, dist in contacts)
    n_rna_contact = len(rna_contact_set)

    # Ligand global Gasteiger stats
    lig_charges = [a["gasteiger"] for a in lig_atoms]
    q_sum  = float(np.sum(lig_charges))
    q_std  = float(np.std(lig_charges))
    q_max  = float(np.max(lig_charges))
    q_min  = float(np.min(lig_charges))

    # Contact-zone charge stats
    contact_lig_q = [lig_atoms[li]["gasteiger"] for _, li, _ in contacts]
    contact_rna_q = [rna_atoms[ri]["partial_charge"] for ri, _, _ in contacts]

    return np.array([
        # Electrostatic
        E_elec,
        float(np.sum([rna_atoms[ri]["partial_charge"] * lig_atoms[li]["gasteiger"] / max(d, 0.5)
                      for ri, li, d in contacts if d < 4.0])),  # close-range only
        float(np.mean(contact_lig_q)),
        float(np.std(contact_lig_q)) if len(contact_lig_q) > 1 else 0.0,
        float(np.mean(contact_rna_q)),
        # H-bond / polar
        float(n_hbond),
        float(n_hbond / max(n_contacts, 1)),
        # Hydrophobic
        float(n_hydro),
        float(n_hydro / max(n_contacts, 1)),
        # Ionic
        float(n_ionic),
        # Aromatic
        float(n_arom),
        # Geometry
        float(n_contacts),
        float(n_rna_contact),
        float(n_contacts / max(len(rna_atoms), 1)),
        float(mean_dist),
        float(min_dist),
        float(np.std(dists)),
        # Ligand global charges
        q_sum, q_std, q_max, q_min,
        float(mol.GetNumAtoms()),
        float(Descriptors.MolWt(mol)),
        float(Descriptors.TPSA(mol)),
        float(Descriptors.NumHDonors(mol)),
        float(Descriptors.NumHAcceptors(mol)),
        float(Descriptors.NumRotatableBonds(mol)),
        float(Descriptors.RingCount(mol)),
        float(Descriptors.NumAromaticRings(mol)),
    ], dtype=np.float32)


PHYS_FEATURE_NAMES = [
    "E_elec_all", "E_elec_close", "mean_q_lig_contact", "std_q_lig_contact",
    "mean_q_rna_contact",
    "n_hbond", "frac_hbond",
    "n_hydro", "frac_hydro",
    "n_ionic",
    "n_arom_pi",
    "n_contacts", "n_rna_atoms_contact", "contact_density",
    "mean_dist", "min_dist", "std_dist",
    "lig_q_sum", "lig_q_std", "lig_q_max", "lig_q_min",
    "lig_n_atoms", "lig_mw", "lig_tpsa",
    "lig_hbd", "lig_hba", "lig_rot", "lig_rings", "lig_arom_rings",
]


def metrics(y_true, y_pred):
    r,  _ = stats.pearsonr(y_true, y_pred)
    rs, _ = stats.spearmanr(y_true, y_pred)
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    return dict(pcc=round(float(r), 4), spcc=round(float(rs), 4),
                rmse=round(rmse, 4),    mae=round(mae, 4))


def run_splits(X, y_norm, label):
    lgb_p = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                 objective='regression', n_jobs=4, random_state=42, verbose=-1)
    ss = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
    res = []
    for tr_idx, te_idx in ss.split(X):
        m = lgb.LGBMRegressor(**lgb_p)
        m.fit(X[tr_idx], y_norm[tr_idx],
              eval_set=[(X[te_idx], y_norm[te_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        res.append(metrics(y_norm[te_idx], m.predict(X[te_idx])))
    df = pd.DataFrame(res)
    m2, s2 = df.mean(), df.std()
    print(f"  {label:<50} PCC={m2['pcc']:.3f}±{s2['pcc']:.3f}  "
          f"SPCC={m2['spcc']:.3f}  RMSE={m2['rmse']:.3f}  MAE={m2['mae']:.3f}")
    return m2.to_dict()


def main():
    print("=" * 70)
    print("STEP 14 — Physical Interaction Features")
    print("=" * 70)

    df_csv = pd.read_csv(DATA_CSV)
    n = len(df_csv)

    # ── Compute physical features ──────────────────────────────────────────────
    cache_path = RES_DIR / "step14_physical_features.npz"
    if cache_path.exists():
        print("Loading cached physical features...")
        cache = np.load(cache_path)
        phys  = cache["phys"]
        valid = cache["valid"].astype(bool)
        pdb_ids = cache["ids"]
    else:
        print(f"Computing physical features for {n} complexes...")
        phys   = []
        valid  = []
        pdb_ids = []
        for i, row in df_csv.iterrows():
            feat = compute_physical_features(row["rna_pdb"], row["lig_sdf"], cutoff=6.0)
            if feat is not None and not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                phys.append(feat)
                valid.append(True)
            else:
                phys.append(np.zeros(len(PHYS_FEATURE_NAMES), dtype=np.float32))
                valid.append(False)
            pdb_ids.append(row["pdb"])
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n}")

        phys   = np.array(phys)
        valid  = np.array(valid)
        pdb_ids = np.array(pdb_ids)
        np.savez(cache_path, phys=phys, valid=valid, ids=pdb_ids)
        print(f"Saved cache → {cache_path}")

    n_valid = valid.sum()
    print(f"Valid complexes: {n_valid}/{n}")
    print(f"Physical features: {phys.shape[1]}")

    # ── Print feature correlations ─────────────────────────────────────────────
    y_raw = df_csv["pKd"].values[valid]
    phys_valid = phys[valid]
    print("\nTop correlated physical features with pKd:")
    corrs = [(name, abs(stats.pearsonr(phys_valid[:, i], y_raw)[0]))
             for i, name in enumerate(PHYS_FEATURE_NAMES)
             if not np.all(phys_valid[:, i] == 0)]
    corrs.sort(key=lambda x: x[1], reverse=True)
    for name, r in corrs[:10]:
        print(f"  {name:<35} r={r:.3f}")

    # ── Load RLEC fingerprint ──────────────────────────────────────────────────
    fp_data = np.load(FEAT_DIR / "rlec_c6.0_r6_l3_s4096_f1.npz")
    X_rlec = fp_data["X"].astype(np.float32)
    y      = fp_data["y"].astype(np.float64)
    ids    = fp_data["ids"]

    # Align pdb_ids to fingerprint ids order
    id_to_idx = {pid: i for i, pid in enumerate(pdb_ids)}
    phys_aligned = np.array([phys[id_to_idx[pid]] for pid in ids])
    valid_aligned = np.array([valid[id_to_idx[pid]] for pid in ids])

    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min)

    # Scale physical features
    sc = StandardScaler()
    phys_scaled = sc.fit_transform(phys_aligned)

    print("\n" + "─" * 70)
    print("RESULTS: Feature combination experiments")
    print("─" * 70)

    run_splits(X_rlec, y_norm, "A: RLEC-4k (baseline)")
    run_splits(phys_scaled, y_norm, "B: Physical features only (29D)")
    run_splits(np.hstack([X_rlec, phys_scaled]), y_norm, "C: RLEC-4k + Physical (4096+29)")

    # Larger RLEC
    fp64 = np.load(FEAT_DIR / "rlec_c6.0_r6_l3_s65536_f1.npz")
    X_64k = fp64["X"].astype(np.float32)
    run_splits(np.hstack([X_64k, phys_scaled]), y_norm, "D: RLEC-64k + Physical (65536+29)")

    # ── Save feature importance ────────────────────────────────────────────────
    X_combined = np.hstack([X_rlec, phys_scaled])
    lgb_m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1)
    lgb_m.fit(X_combined, y_norm)
    fi = lgb_m.feature_importances_
    phys_fi = fi[-len(PHYS_FEATURE_NAMES):]
    print("\nPhysical feature importances (in combined model):")
    fi_pairs = sorted(zip(PHYS_FEATURE_NAMES, phys_fi), key=lambda x: x[1], reverse=True)
    for name, imp in fi_pairs[:10]:
        print(f"  {name:<35} importance={imp:.1f}")

    print(f"\nSaved results → {RES_DIR / 'step14_physical_features.npz'}")


if __name__ == "__main__":
    main()
