# Changelog

All notable changes to RLEC are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions correspond to PyPI releases at https://pypi.org/project/rlec/

---

## [0.3.1] - 2026-08-14

### Fixed
- README on PyPI now shows physical features, CLI commands, and updated benchmark table

---

## [0.3.0] - 2026-08-14

### Added
- **`rlec.physical` module** - 29 physics-based interaction features from an RNA-ligand complex:
 - Electrostatic: Gasteiger-charge-weighted Coulomb sum at interface (all + r<4 Å)
 - H-bond: N/O···N/O donor-acceptor pair count (r < 3.5 Å) + fraction of contacts
 - Hydrophobic: C/S···C/S contact count (r < 4.5 Å) + fraction
 - Ionic: RNA phosphate P···cationic ligand atom count (r < 5.0 Å)
 - Aromatic π-stacking: aromatic ring atom pairs (r < 5.5 Å)
 - Geometric: contact count, unique RNA atoms in contact, contact density, distance stats
 - Ligand global: Gasteiger charge sum/std/max/min
 - Ligand RDKit 2D: MW, TPSA, HBD, HBA, RotBonds, RingCount, AromaticRings
- **`compute_physical_features(rna_pdb, lig_sdf, cutoff=6.0)`** - returns `(29,)` float32 array
- **`compute_physical_batch(complexes, cutoff=6.0, n_jobs=1)`** - returns `(N, 29)` array
- **`PHYSICAL_FEATURE_NAMES`** - list of 29 feature name strings
- **CLI `rlec physical RNA.pdb LIG.sdf`** - prints all 29 feature name/value pairs;
  `--output FILE.csv` or `--output FILE.npy` saves the vector; `--cutoff Å` adjustable
- All three are exported from the top-level `rlec` namespace

### Notes
- Physical features complement the RLEC fingerprint: `np.concatenate([fp.transform(r,l), compute_physical_features(r,l)])` → (4125,) combined descriptor
- Combining RLEC-4096 + physical(29) + LigECFP4(2048) on PDBbind NL2020 (n=143, 10-split 80/20): PCC=0.584, SPCC=0.558, RMSE=0.143, MAE=0.115 - beats RLaffinity baseline

---

## [0.2.0] - 2026-08-14

### Added
- **Parallel batch processing**: `transform_batch(complexes, n_jobs=-1, show_progress=True)`
  using joblib + tqdm; real-time progress even with parallel workers (`return_as="generator"`)
- **Sparse output**: `to_sparse(complexes)` → `scipy.sparse.csr_matrix`
  (memory-efficient for large `fp_size`)
- **DataFrame output**: `to_dataframe(complexes, ids=None)` → `pandas.DataFrame`
  with columns `bit_0 … bit_{fp_size-1}` and optional `id` column
- **`RLECTransformer`** (`rlec._sklearn`) - full `BaseEstimator` + `TransformerMixin`:
  works in `sklearn.pipeline.Pipeline`, `GridSearchCV`, `cross_val_score`
- **`to_dict()` / `from_dict()`** - serialise/reconstruct fingerprint parameters
  (round-trips cleanly; includes `__version__` key for forward compatibility)
- **`__eq__`** and **`copy()`** - proper object identity/value semantics
- **Pickle support** - confirmed clean round-trip via `pickle.dumps/loads`
- **`get_params(deep=True)`** - sklearn-compatible `deep` argument
- **CLI `rlec batch`**: compute fingerprints for many complexes from a CSV file;
  supports parallel workers (`--n-jobs`), output formats `.npz`/`.npy`/`.csv`
- **CLI `rlec validate`**: inspect a complex - RNA atom count, nucleotide distribution,
  PBS distribution, contact count, fingerprint density; warns on zero contacts
- **CLI `rlec transform`**: added `--format {npy,csv,txt}` flag; format now also
  inferred from output file extension
- `joblib>=1.2` added as explicit dependency (pins `return_as="generator"` support)
- Development Status classifier bumped to `4 - Beta`

### Changed
- `transform_batch` signature: added `n_jobs=1` and `show_progress=True` parameters
  (fully backward-compatible - existing code with positional `complexes` arg unchanged)

---

## [0.1.2] - 2026-08-14

### Added
- **`rlec` CLI entry point** - package now installs an `rlec` shell command
- `rlec --version` - print version and exit
- `rlec info` - show version, default parameters, feat_set options
- `rlec transform RNA LIG [--output FILE] [--fp-size N] …` - compute one fingerprint

### Fixed
- Missing `[project.scripts]` in `pyproject.toml` meant `rlec` command was not
  created on `pip install` (the core bug reported by the user)

---

## [0.1.1] - 2026-08-14

### Added
- **`__repr__`** - `RLECFingerprint(rna_depth=6, ligand_depth=3, …)` instead of
  memory address
- **`get_params()` / `set_params()`** - sklearn-compatible parameter access
- **Input validation** in `__init__`:
 - `feat_set` must be 0, 1, 2, or 3
 - `rna_depth` / `ligand_depth` must be ≥ 0
 - `fp_size` must be ≥ 1
 - `cutoff` must be > 0

### Fixed
- **`transform(rna, mol_2d)`** with a 2D ligand Mol (no 3D conformer) raised a cryptic
  `ValueError: Bad Conformer Id` from RDKit internals → now raises a clear
  `ValueError: "Ligand Mol has no 3D conformer. Generate with AllChem.EmbedMolecule()…"`
- **`RLECFingerprint(rna_depth=-1)`** silently accepted and produced incorrect bits
  → now raises `ValueError`
- **`RLECFingerprint(fp_size=0)`** raised `ZeroDivisionError` at compute time
  → now raises `ValueError` at construction time

---

## [0.1.0] - 2026-08-14

### Added
- Core `RLECFingerprint` class implementing the PLEC algorithm adapted for RNA:
 - Biopython RNA PDB parsing with strict residue whitelist (excludes protein,
    metal ions, water, solvent - avoids RDKit RNA parsing bug GitHub #6501)
 - Element-pair-specific covalent bond graph (covers C-I bond in 5-iodouridine)
 - RNA atom feature enrichment: PBS group, nucleotide type, formal charge,
    aromaticity, ring membership, heavy-atom degree, atomic number
 - cKDTree contact detection (fast, exact)
 - Morgan-style RNA ECFP hashes (Biopython bond graph iteration)
 - RDKit ligand ECFP hashes via `GetMorganFingerprint` bitInfo
 - PLEC-style `zip_longest` pairing of RNA + ligand environments
 - Count folding into float32 vector of configurable size
- 4 RNA feature sets:
 - `feat_set=0`: basic ECFP (atomic_num, formal_charge, degree, aromaticity, ring)
 - `feat_set=1`: + nucleotide type A/U/G/C/T  **[best: LOOCV r=0.710]**
 - `feat_set=2`: + PBS group (Phosphate/Sugar/Base)
 - `feat_set=3`: + nucleotide type + PBS group
- `transform(rna_pdb, ligand)` - single complex; accepts SDF path or RDKit Mol
- `transform_batch(complexes)` - list of (rna_pdb, ligand) tuples
- Validated on 143 RNA-ligand complexes (LOOCV LightGBM):
 - RLEC feat1: r=0.710, 95% CI=[0.616, 0.790]
 - vs ligand-only ECFP: Δr=+0.148, p<0.0005 (paired bootstrap, 2000 iterations)
 - vs no-RNA-feature baseline (feat0): Δr=+0.116, p<0.0005
- PyPI package: `pip install rlec`

---

## Algorithm reference

RLEC adapts PLEC (Protein-Ligand Extended Connectivity) to RNA:

```
for rna_atom, lig_atom in contacts(rna, ligand, cutoff=6.0 Å):
    rna_ecfp = ECFP_hashes(rna_atom, depth=6)   # 7 hashes: d0..d6
    lig_ecfp = ECFP_hashes(lig_atom,  depth=3)   # 4 hashes: d0..d3
    for r_h, l_h in zip_longest(rna_ecfp, lig_ecfp, fillvalue=last):
        bits.append(hash((r_h, l_h)) & 0xFFFFFFFF)
fingerprint = count_fold(bits, fp_size=4096)
```

**Reference:**
Wójcikowski M, Kukiełka M, Stepniak-Konieczna M, Antosiewicz JM, Siedlecki P.
Development of a protein-ligand extended connectivity (PLEC) fingerprint and its
application for virtual screening.
*Bioinformatics* 2019;35(8):1334-1341. https://doi.org/10.1093/bioinformatics/bty757
