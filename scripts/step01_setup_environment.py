"""
STEP 01 — Environment Setup and Dependency Verification
RLEC: RNA-Ligand Extended Connectivity Fingerprint Project

Checks all required packages, verifies RDKit can parse RNA PDB structures,
verifies ODDT is available, and saves environment info to logs/.
"""

import sys
import os
import json
import importlib
import subprocess
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("tqdm", "tqdm"),
    ("requests", "requests"),
    ("Bio", "biopython"),
    ("rdkit", "rdkit"),
]

OPTIONAL_PACKAGES = [
    ("oddt", "oddt"),
    ("prolif", "prolif"),
]

def check_package(import_name, pip_name):
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        return {"status": "ok", "version": version, "pip_name": pip_name}
    except ImportError:
        return {"status": "MISSING", "version": None, "pip_name": pip_name}

def install_missing(pip_name):
    print(f"  Installing {pip_name}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name, "-q"],
        capture_output=True, text=True
    )
    return result.returncode == 0

def verify_rdkit_rna():
    """Test RDKit can parse a minimal RNA PDB block."""
    try:
        from rdkit import Chem
        # Minimal RNA dinucleotide PDB block (AU, synthetic)
        pdb_block = """ATOM      1  P     A A   1      -0.956   4.671   0.000  1.00  0.00           P
ATOM      2  O1P   A A   1      -1.744   5.736   0.705  1.00  0.00           O
ATOM      3  O2P   A A   1      -1.528   3.272  -0.144  1.00  0.00           O
ATOM      4  O5'   A A   1       0.584   4.904  -0.100  1.00  0.00           O
ATOM      5  C5'   A A   1       1.424   3.774  -0.018  1.00  0.00           C
ATOM      6  C4'   A A   1       2.859   4.176   0.222  1.00  0.00           C
ATOM      7  O4'   A A   1       3.605   3.069   0.734  1.00  0.00           O
ATOM      8  C3'   A A   1       3.563   4.793  -0.990  1.00  0.00           C
ATOM      9  O3'   A A   1       3.332   6.196  -1.035  1.00  0.00           O
ATOM     10  C2'   A A   1       5.026   4.493  -0.710  1.00  0.00           C
ATOM     11  O2'   A A   1       5.682   5.741  -0.494  1.00  0.00           O
ATOM     12  C1'   A A   1       4.942   3.640   0.565  1.00  0.00           C
ATOM     13  N9    A A   1       5.321   2.226   0.413  1.00  0.00           N
ATOM     14  C8    A A   1       4.502   1.211   0.952  1.00  0.00           C
ATOM     15  N7    A A   1       5.053   0.056   0.645  1.00  0.00           N
ATOM     16  C5    A A   1       6.310   0.347  -0.059  1.00  0.00           C
ATOM     17  C6    A A   1       7.351  -0.545  -0.499  1.00  0.00           C
ATOM     18  N6    A A   1       7.197  -1.861  -0.234  1.00  0.00           N
ATOM     19  N1    A A   1       8.531  -0.071  -1.170  1.00  0.00           N
ATOM     20  C2    A A   1       8.689   1.258  -1.429  1.00  0.00           C
ATOM     21  N3    A A   1       7.735   2.149  -1.022  1.00  0.00           N
ATOM     22  C4    A A   1       6.554   1.669  -0.348  1.00  0.00           C
END
"""
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
        if mol is None:
            return {"status": "FAIL", "error": "MolFromPDBBlock returned None"}
        n_atoms = mol.GetNumAtoms()
        return {"status": "ok", "atoms_parsed": n_atoms}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

def verify_oddt():
    try:
        import oddt
        from oddt import toolkit
        return {"status": "ok", "version": getattr(oddt, "__version__", "unknown")}
    except ImportError:
        return {"status": "not installed (optional)"}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

def verify_biopython_pdb():
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

def main():
    print("=" * 60)
    print("RLEC Step 01 — Environment Setup")
    print("=" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "packages": {},
        "optional_packages": {},
        "rdkit_rna_test": None,
        "oddt_test": None,
        "biopython_pdb_test": None,
        "overall_status": "PASS",
        "missing_required": [],
    }

    # --- Required packages ---
    print("\n[1] Checking required packages...")
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        result = check_package(import_name, pip_name)
        report["packages"][pip_name] = result
        if result["status"] == "MISSING":
            print(f"  MISSING: {pip_name}  -> attempting install")
            ok = install_missing(pip_name)
            result2 = check_package(import_name, pip_name)
            report["packages"][pip_name] = result2
            if result2["status"] != "ok":
                missing.append(pip_name)
                print(f"  FAILED to install {pip_name}")
            else:
                print(f"  Installed {pip_name} v{result2['version']}")
        else:
            print(f"  OK  {pip_name} v{result['version']}")

    # --- Optional packages ---
    print("\n[2] Checking optional packages...")
    for import_name, pip_name in OPTIONAL_PACKAGES:
        result = check_package(import_name, pip_name)
        report["optional_packages"][pip_name] = result
        status = result["status"]
        version = result.get("version", "")
        print(f"  {'OK' if status == 'ok' else 'optional/missing'}  {pip_name} {version}")

    # --- RDKit RNA parse test ---
    print("\n[3] Testing RDKit RNA PDB parsing...")
    rdkit_test = verify_rdkit_rna()
    report["rdkit_rna_test"] = rdkit_test
    print(f"  RDKit RNA test: {rdkit_test}")

    # --- ODDT test ---
    print("\n[4] Testing ODDT...")
    oddt_test = verify_oddt()
    report["oddt_test"] = oddt_test
    print(f"  ODDT: {oddt_test}")

    # --- BioPython PDB ---
    print("\n[5] Testing BioPython PDB parser...")
    bp_test = verify_biopython_pdb()
    report["biopython_pdb_test"] = bp_test
    print(f"  BioPython PDB: {bp_test}")

    # --- Summary ---
    if missing:
        report["overall_status"] = "FAIL"
        report["missing_required"] = missing
        print(f"\n[FAIL] Missing required packages: {missing}")
    else:
        print("\n[PASS] All required packages available.")

    # --- Save report ---
    log_path = LOG_DIR / "step01_environment.json"
    with open(log_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nEnvironment report saved: {log_path}")

    return 0 if not missing else 1

if __name__ == "__main__":
    sys.exit(main())
