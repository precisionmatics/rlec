# RLEC Research Findings — Deep Literature Survey
# Date: 2026-08-13

## Key Databases for RNA-Ligand Affinity Data
- **PDBbind RNA subset v2024**: 234 complexes, 3D structures + Kd/Ki/IC50 (best quality)
- **R-SIM**: 2,501 pairs; 1,524 with Kd values (used by RSAPred, DeepRSMA)
- **RiboBind (2025)**: 1,591 PDB complexes (arXiv:2503.17007)
- **RNALID**: 358 manually curated (RNA Biology 2023)
- Our dataset: 143 complexes from PDFL-RNA — already collected and validated

## Best Existing RNA-Ligand Affinity Predictors
| Method | Type | Pearson r | Dataset | Year |
|--------|------|-----------|---------|------|
| RSAPred | Sequence + linear | 0.83 avg, 0.897 viral | R-SIM Kd | 2024 |
| DeepRSMA | GNN + Transformer | 0.784 CV, 0.490 blind | R-SIM | 2024 |
| RLaffinity | 3D-CNN | ~0.83* (contested) | R-SIM | 2024 |
| RLASIF | Geometric DL | +10% vs 2nd best | PDBbind NL2020 | 2025 |

*RLEC target: beat RSAPred (r=0.83) on same/similar 143-complex set*

## Existing RNA Interaction Fingerprints
- **fingeRNAt**: 9 explicit interaction types (H-bond, stacking, halogen, cation-anion, Pi-π, water/metal-mediated); PBS resolution (Phosphate/Base/Sugar). Ref: Szulc et al., PLOS Comp Biol 2022
- **ProLIF**: SMARTS-based, works with RNA, no RNA-specific atom features
- **PLIP 2021**: Extended to RNA/DNA, detects H-bonds, π-stacking, hydrophobic, etc.
- NONE of these use paired ECFP environments (PLEC-style) — RLEC is novel

## RNA-Ligand Interaction Statistics (from Guillen-Chable, RSC Med Chem 2020)
- Stacking: 34.8% (vs 20.2% in protein-ligand) — most dominant in RNA!
- H-bonding: 34.4%
- Hydrophobic: 17.8% (vs 47.2% in protein — much less in RNA)
- Electrostatic (cation-π + salt bridges): 5.8%
- Adenine stacks most (A > G > U > C)

## Critical RNA Atom Features for RLEC
Standard ECFP features (atomic number, charge, aromaticity, ring, #H, #heavy neighbors)
PLUS RNA-specific:
1. Nucleotide type: A/U/G/C (one-hot)
2. PBS group: Phosphate / Base / Sugar (one-hot) — from fingeRNAt
3. Is 2'-OH oxygen (bool) — unique RNA vs DNA marker
4. Base pair status: unpaired / WC-paired / non-WC (from DSSR)
5. Secondary structure element: stem/loop/bulge/junction

## RDKit PDB Parsing Issues (GitHub Issue #6501) — CRITICAL
- RDKit does NOT comply with PDB spec for bond assignment
- proximityBonding=True: spurious bonds (nucleobases 3.4 Å apart → fake bonds)
- proximityBonding=False: missing bonds in standard residues
- SOLUTION: Parse RNA via Biopython (atom coordinates only), get ligand from SDF (RDKit handles perfectly), use custom distance-based contact detection (KD-tree)

## Key Papers for Citations
1. Wójcikowski et al., Bioinformatics 35(8):1334 (2019) — PLEC
2. Szulc et al., PLOS Comp Biol 18(6):e1009783 (2022) — fingeRNAt
3. Szulc et al., Briefings Bioinformatics 24(4):bbad187 (2023) — RNA ML screening
4. Krishnan et al., Briefings Bioinformatics 25(2):bbae002 (2024) — RSAPred
5. Hu et al., Bioinformatics 40(12):btae678 (2024) — DeepRSMA
6. Oliver et al., Nature Communications 16 (2025) — RNAmigos2
7. Guillen-Chable, RSC Med Chem 11:802 (2020) — interaction statistics
8. Bouysset & Fiorucci, J Cheminformatics 13:72 (2021) — ProLIF
9. Adasme et al., NAR 49(W1):W530 (2021) — PLIP 2021
10. Sun & Gao, Bioinformatics 40(4):btae155 (2024) — RLaffinity 3D-CNN
