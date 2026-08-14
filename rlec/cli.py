"""
RLEC command-line interface.

Commands:
    rlec --version
    rlec info
    rlec transform RNA.pdb LIGAND.sdf [options]
    rlec batch dataset.csv --rna-col COL --lig-col COL [options]
    rlec validate RNA.pdb LIGAND.sdf [options]
"""

from __future__ import annotations

import argparse
import sys


# ── Shared FP parameter arguments ─────────────────────────────────────────────

def _add_fp_args(parser: argparse.ArgumentParser) -> None:
    """Add the five RLECFingerprint parameters to a subparser."""
    parser.add_argument("--rna-depth",  type=int,   default=6,    dest="rna_depth",
                        metavar="N", help="Morgan depth for RNA atoms (default: 6)")
    parser.add_argument("--lig-depth",  type=int,   default=3,    dest="lig_depth",
                        metavar="N", help="Morgan depth for ligand atoms (default: 3)")
    parser.add_argument("--fp-size",    type=int,   default=4096, dest="fp_size",
                        metavar="N", help="Fingerprint vector size (default: 4096)")
    parser.add_argument("--feat-set",   type=int,   default=1,    dest="feat_set",
                        choices=[0, 1, 2, 3], metavar="{0,1,2,3}",
                        help="RNA feature set: 0=basic, 1=+nuc_type (best), "
                             "2=+PBS, 3=+both (default: 1)")
    parser.add_argument("--cutoff",     type=float, default=6.0,
                        metavar="Å",  help="Contact distance cutoff in Å (default: 6.0)")


def _make_fp(args) -> "RLECFingerprint":
    from rlec import RLECFingerprint
    return RLECFingerprint(
        rna_depth=args.rna_depth,
        ligand_depth=args.lig_depth,
        fp_size=args.fp_size,
        feat_set=args.feat_set,
        cutoff=args.cutoff,
    )


# ── rlec info ─────────────────────────────────────────────────────────────────

def cmd_info(args) -> None:
    from rlec import __version__, RLECFingerprint
    fp = RLECFingerprint()
    print(f"rlec {__version__}")
    print(f"Default fingerprint: {fp!r}")
    print()
    print("feat_set options:")
    print("  0 — basic ECFP  (atomic_num, formal_charge, degree, aromaticity, ring)")
    print("  1 — + nucleotide type A/U/G/C/T  [BEST — LOOCV r=0.710 on 143 complexes]")
    print("  2 — + PBS group (Phosphate / Sugar / Base)")
    print("  3 — + nucleotide type + PBS group")
    print()
    print("Performance (LOOCV LightGBM, n=143):")
    print("  Ligand-only ECFP : r=0.562")
    print("  RNA-only ECFP    : r=0.594")
    print("  RLEC feat0       : r=0.593")
    print("  RLEC feat1       : r=0.710  ← recommended")
    print("  95% CI           : [0.616, 0.790]")
    print("  vs ligand-only   : Δr=+0.148, p<0.0005 (paired bootstrap)")


# ── rlec transform ────────────────────────────────────────────────────────────

def cmd_transform(args) -> None:
    import numpy as np
    fp = _make_fp(args)
    vec = fp.transform(args.rna, args.ligand)
    nonzero = int((vec > 0).sum())

    fmt = args.format
    if args.output:
        # infer format from extension if not specified
        if fmt is None:
            ext = args.output.rsplit(".", 1)[-1].lower()
            fmt = {"npy": "npy", "csv": "csv", "txt": "txt"}.get(ext, "npy")
        if fmt == "npy":
            np.save(args.output, vec)
        elif fmt == "csv":
            import pandas as pd
            cols = [f"bit_{i}" for i in range(fp.fp_size)]
            pd.DataFrame([vec], columns=cols).to_csv(args.output, index=False)
        elif fmt == "txt":
            with open(args.output, "w") as fh:
                fh.write(" ".join(str(int(x)) for x in vec) + "\n")
        print(f"Saved {args.output}  shape={vec.shape}  nonzero={nonzero}  sum={vec.sum():.0f}")
    else:
        if fp.fp_size <= 256 or fmt == "txt":
            print(" ".join(str(int(x)) for x in vec))
        else:
            print(f"shape=({fp.fp_size},)  nonzero={nonzero}  sum={vec.sum():.0f}")
            print("Tip: use --output FILE.npy to save the full vector.")


# ── rlec batch ────────────────────────────────────────────────────────────────

def cmd_batch(args) -> None:
    import numpy as np
    import pandas as pd
    from rlec import RLECFingerprint

    df = pd.read_csv(args.input)

    for col in (args.rna_col, args.lig_col):
        if col not in df.columns:
            print(f"ERROR: column '{col}' not found. Available: {list(df.columns)}",
                  file=sys.stderr)
            sys.exit(1)

    complexes = list(zip(df[args.rna_col], df[args.lig_col]))
    ids = list(df[args.id_col].astype(str)) if args.id_col else [str(i) for i in range(len(df))]

    fp = _make_fp(args)
    print(f"Computing RLEC for {len(complexes)} complexes "
          f"(fp_size={fp.fp_size}, n_jobs={args.n_jobs})...")

    X = fp.transform_batch(complexes, n_jobs=args.n_jobs,
                           show_progress=not args.no_progress)

    if args.output:
        ext = args.output.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            out_df = fp.to_dataframe(complexes, ids=ids, show_progress=False)
            out_df.to_csv(args.output, index=False)
        elif ext == "npy":
            np.save(args.output, X)
        else:  # default: npz
            np.savez_compressed(args.output, fingerprints=X, ids=np.array(ids))
        print(f"Saved {args.output}  shape={X.shape}")
    else:
        print(f"shape={X.shape}  nonzero/row={(X > 0).sum(1).mean():.1f}  "
              f"sum/row={X.sum(1).mean():.1f}")
        print("Tip: use --output FILE.npz (or .npy/.csv) to save the matrix.")


# ── rlec validate ─────────────────────────────────────────────────────────────

def cmd_validate(args) -> None:
    from rlec.fingerprint import (
        _parse_rna_atoms, _build_bond_graph, _enrich_rna_atoms,
        _mol_from_sdf, _lig_atoms_from_mol, _detect_contacts,
    )
    fp = _make_fp(args)

    print(f"{'─'*50}")
    print(f"  RNA PDB : {args.rna}")
    print(f"  Ligand  : {args.ligand}")
    print(f"  Cutoff  : {fp.cutoff} Å   feat_set={fp.feat_set}")
    print(f"{'─'*50}")

    rna_atoms = _parse_rna_atoms(args.rna)
    if not rna_atoms:
        print("ERROR: No RNA nucleotide atoms found. Check the PDB file.", file=sys.stderr)
        sys.exit(1)
    print(f"  RNA atoms        : {len(rna_atoms)}")

    graph = _build_bond_graph(rna_atoms)
    _enrich_rna_atoms(rna_atoms, graph)

    # nucleotide distribution
    from collections import Counter
    nuc_names = {0: "A", 1: "U", 2: "G", 3: "C", 4: "T", 5: "mod"}
    nuc_counts = Counter(nuc_names[a["nuc_type"]] for a in rna_atoms)
    pbs_names = {0: "P", 1: "S", 2: "B"}
    pbs_counts = Counter(pbs_names[a["pbs"]] for a in rna_atoms)
    print(f"  Nucleotide dist  : {dict(sorted(nuc_counts.items()))}")
    print(f"  PBS dist         : P={pbs_counts['P']}  S={pbs_counts['S']}  B={pbs_counts['B']}")

    mol = _mol_from_sdf(args.ligand)
    if mol is None:
        print("ERROR: Could not parse ligand SDF.", file=sys.stderr)
        sys.exit(1)
    lig_atoms = _lig_atoms_from_mol(mol)
    print(f"  Ligand atoms     : {len(lig_atoms)}")

    contacts = _detect_contacts(rna_atoms, lig_atoms, fp.cutoff)
    print(f"  Contacts (<{fp.cutoff}Å) : {len(contacts)}")

    if not contacts:
        print()
        print("  WARNING: No contacts found — fingerprint will be all-zero.")
        print("           Check that the PDB and SDF describe the same complex.")
        return

    vec = fp.transform(args.rna, args.ligand)
    nonzero = int((vec > 0).sum())
    density = 100.0 * nonzero / fp.fp_size
    print(f"  Nonzero bits     : {nonzero} / {fp.fp_size}  ({density:.2f}% density)")
    print(f"  Total count      : {int(vec.sum())}")
    print(f"  Max count        : {int(vec.max())}")
    print(f"{'─'*50}")
    print("  OK — fingerprint computed successfully.")


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rlec",
        description="RNA-Ligand Extended Connectivity (RLEC) Fingerprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  rlec info\n"
            "  rlec transform rna.pdb lig.sdf --output vec.npy\n"
            "  rlec batch data.csv --rna-col rna_path --lig-col lig_path -o feats.npz --n-jobs -1\n"
            "  rlec validate rna.pdb lig.sdf\n"
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__import__('rlec').__version__}",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = False

    # ── info ──
    p_info = sub.add_parser("info", help="Show version, defaults, and performance")
    p_info.set_defaults(func=cmd_info)

    # ── transform ──
    p_tr = sub.add_parser(
        "transform",
        help="Compute RLEC fingerprint for a single complex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_tr.add_argument("rna",    help="RNA PDB file path")
    p_tr.add_argument("ligand", help="Ligand SDF file path")
    p_tr.add_argument("-o", "--output", default=None, metavar="FILE",
                      help="Output file (.npy / .csv / .txt)")
    p_tr.add_argument("--format", choices=["npy", "csv", "txt"], default=None,
                      help="Output format (inferred from --output extension if omitted)")
    _add_fp_args(p_tr)
    p_tr.set_defaults(func=cmd_transform)

    # ── batch ──
    p_bt = sub.add_parser(
        "batch",
        help="Compute RLEC fingerprints for many complexes from a CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Reads a CSV file with columns pointing to PDB/SDF paths,\n"
            "computes RLEC for every row, and saves the feature matrix.\n\n"
            "Output formats: .npz (default, compressed), .npy, .csv"
        ),
    )
    p_bt.add_argument("input",      help="Input CSV file")
    p_bt.add_argument("--rna-col",  required=True, dest="rna_col",
                      metavar="COL", help="CSV column with RNA PDB paths")
    p_bt.add_argument("--lig-col",  required=True, dest="lig_col",
                      metavar="COL", help="CSV column with ligand SDF paths")
    p_bt.add_argument("--id-col",   default=None, dest="id_col",
                      metavar="COL", help="CSV column to use as row IDs (optional)")
    p_bt.add_argument("-o", "--output", default=None, metavar="FILE",
                      help="Output file (.npz / .npy / .csv)")
    p_bt.add_argument("--n-jobs",   type=int, default=1, dest="n_jobs",
                      metavar="N",  help="Parallel workers (-1 = all CPUs, default: 1)")
    p_bt.add_argument("--no-progress", action="store_true",
                      help="Suppress tqdm progress bar")
    _add_fp_args(p_bt)
    p_bt.set_defaults(func=cmd_batch)

    # ── validate ──
    p_vl = sub.add_parser(
        "validate",
        help="Inspect a complex: atom counts, contacts, fingerprint density",
    )
    p_vl.add_argument("rna",    help="RNA PDB file path")
    p_vl.add_argument("ligand", help="Ligand SDF file path")
    _add_fp_args(p_vl)
    p_vl.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
