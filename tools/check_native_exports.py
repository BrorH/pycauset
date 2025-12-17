"""Dev tool: detect drift between Python expectations and native exports.

Run from repo root:
  - `python tools/check_native_exports.py`

This imports the *in-tree* Python package (`python/pycauset`) and then verifies
that `pycauset._pycauset` exports the symbols that the public Python surface
relies on.

Exit code:
  - 0: OK
  - 1: Missing required exports or smoke test failed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


def _has(native: object, name: str) -> bool:
    return getattr(native, name, None) is not None


def _check_any(native: object, names: Iterable[str]) -> tuple[bool, str]:
    names = tuple(names)
    ok = any(_has(native, n) for n in names)
    return ok, " or ".join(names)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check pycauset native exports (dev drift check)")
    parser.add_argument(
        "--package-dir",
        default="python",
        help="Path to the in-tree Python package root (default: ./python)",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="Skip runtime smoke tests (only check symbol presence).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat optional exports as required.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    package_dir = (repo_root / args.package_dir).resolve()

    if not package_dir.exists():
        print(f"ERROR: package dir not found: {package_dir}")
        return 1

    sys.path.insert(0, str(package_dir))

    try:
        import pycauset  # noqa: F401
        import pycauset._pycauset as native
    except Exception as exc:
        print("ERROR: failed to import pycauset / pycauset._pycauset")
        print(f"  {type(exc).__name__}: {exc}")
        return 1

    required_anyof: list[tuple[str, tuple[str, ...]]] = [
        ("Float matrix type", ("FloatMatrix",)),
        ("Float32 matrix type", ("Float32Matrix",)),
    ]

    # These are required for the *current* Python surface to function.
    required: dict[str, str] = {
        # Causal set / sprinkling
        "MinkowskiDiamond": "CausalSet default spacetime",
        "sprinkle": "CausalSet sprinkling",
        "make_coordinates": "CausalSet.coordinates",
        # Core types used by factories
        "TriangularBitMatrix": "causal_matrix + causal sprinkling output",
        "DenseBitMatrix": "bool dense matrix support",
        "IntegerMatrix": "int matrix support",
        "TriangularFloatMatrix": "compute_k result",
        "TriangularIntegerMatrix": "triangular int support",
        # Operations invoked by Python wrappers
        "matmul": "pycauset.matmul optimized path",
        "compute_k_matrix": "pycauset.compute_k",
        # Vectors
        "FloatVector": "pycauset.vector default",
        "IntegerVector": "pycauset.vector(int) support",
        "BitVector": "pycauset.vector(bool) support",
        "UnitVector": "unit vectors used by persistence/engine",
        # System
        "MemoryGovernor": "memory governor public API",
    }

    optional: dict[str, str] = {
        "MinkowskiCylinder": "extra spacetime",
        "MinkowskiBox": "extra spacetime",
    }

    missing_required: list[str] = []

    for label, names in required_anyof:
        ok, rendered = _check_any(native, names)
        if not ok:
            missing_required.append(f"{label}: {rendered}")

    for name, reason in required.items():
        if not _has(native, name):
            missing_required.append(f"{name} ({reason})")

    missing_optional: list[str] = []
    for name, reason in optional.items():
        if not _has(native, name):
            missing_optional.append(f"{name} ({reason})")

    if missing_required:
        print("FAIL: missing required native exports:")
        for item in missing_required:
            print(f"  - {item}")
        return 1

    if missing_optional:
        kind = "FAIL" if args.strict else "WARN"
        print(f"{kind}: missing optional native exports:")
        for item in missing_optional:
            print(f"  - {item}")
        if args.strict:
            return 1

    if not args.no_smoke:
        # A few cheap runtime checks to catch signature/name mismatches.
        try:
            # Ensure TriangularBitMatrix.random accepts legacy keyword `p`.
            tbm = native.TriangularBitMatrix.random(5, p=0.5)
            if tbm.size() != 5:
                raise RuntimeError("TriangularBitMatrix.random returned wrong size")

            import pycauset

            cs = pycauset.CausalSet(n=10)
            C = cs.C
            K = pycauset.compute_k(C, 1.0)
            if K.size() != 10:
                raise RuntimeError("compute_k returned wrong size")

            v = pycauset.vector([1, 2, 3])
            if len(v) != 3:
                raise RuntimeError("vector factory returned wrong length")
        except Exception as exc:
            print("FAIL: smoke test failed")
            print(f"  {type(exc).__name__}: {exc}")
            return 1

    print("OK: native exports look consistent with Python expectations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
