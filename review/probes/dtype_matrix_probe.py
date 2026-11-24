"""Probe: dtype x operation grid against NumPy, including save/load roundtrip.

For each dtype the probe builds small random NxM operands, runs each op in both
libraries and compares values *and* result dtype. Output is a compact table;
`ok` = values and dtype match, `VAL` = wrong values, `DT:x` = right values but
dtype x where NumPy gives something else, `ERR` = exception.

Run:  python review/probes/dtype_matrix_probe.py [--verbose]
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings

import numpy as np

import pycauset as pc

warnings.simplefilter("ignore")
VERBOSE = "--verbose" in sys.argv

DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
          "uint64", "float16", "float32", "float64", "complex64", "complex128"]


def rand(dt, shape, rng):
    if dt == "bool":
        return rng.random(shape) < 0.5
    if dt.startswith(("int", "uint")):
        return rng.integers(0, 5, size=shape).astype(dt)
    if dt.startswith("complex"):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
    return rng.standard_normal(shape).astype(dt)


def arr(x):
    if isinstance(x, (bool, int, float, complex, np.generic)):
        return np.asarray(x)
    return pc.to_numpy(x, allow_huge=True)


def save_load(x):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.pycauset")
    pc.save(x, p)
    return pc.load(p)


OPS = {
    "export": (lambda a, b: a, lambda a, b: a),
    "T": (lambda a, b: a.T, lambda a, b: a.T),
    "a+b": (lambda a, b: a + b, lambda a, b: a + b),
    "a-b": (lambda a, b: a - b, lambda a, b: a - b),
    "a*b": (lambda a, b: a * b, lambda a, b: a * b),
    "a*3": (lambda a, b: a * 3, lambda a, b: a * 3),
    "-a": (lambda a, b: -a, lambda a, b: -a),
    "a@bT": (lambda a, b: a @ b.T, lambda a, b: a @ b.T),
    "save/load": (lambda a, b: save_load(a), lambda a, b: a),
}


def main() -> None:
    rng = np.random.default_rng(2)
    shape = (6, 4)
    print("dtype".ljust(11) + "".join(k.ljust(11) for k in OPS))
    problems = []
    for dt in DTYPES:
        A, B = rand(dt, shape, rng), rand(dt, shape, rng)
        row = dt.ljust(11)
        try:
            pA, pB = pc.matrix(A), pc.matrix(B)
        except Exception as e:  # noqa: BLE001
            print(row + f"construct ERR {type(e).__name__}: {str(e)[:80]}")
            continue
        for name, (pf, nf) in OPS.items():
            try:
                with np.errstate(all="ignore"):
                    ref = np.asarray(nf(A, B))
            except Exception:  # noqa: BLE001
                row += "n/a".ljust(11)
                continue
            try:
                got = pf(pA, pB)
                if hasattr(got, "eval"):
                    got = got.eval()
                g = arr(got)
                tol = 1e-2 if dt == "float16" else 1e-5
                if g.shape != ref.shape or not np.allclose(g.astype(np.complex128), ref.astype(np.complex128), rtol=tol, atol=tol):
                    cell = "VAL"
                    problems.append((dt, name, f"got {g.dtype}{g.shape} expected {ref.dtype}{ref.shape}"))
                elif g.dtype != ref.dtype:
                    cell = f"DT:{g.dtype}"
                    problems.append((dt, name, f"dtype {g.dtype} vs numpy {ref.dtype}"))
                else:
                    cell = "ok"
            except Exception as e:  # noqa: BLE001
                cell = "ERR"
                problems.append((dt, name, f"{type(e).__name__}: {str(e)[:110]}"))
            row += cell.ljust(11)
        print(row)
    if VERBOSE:
        print()
        for p in problems:
            print(" | ".join(p))


if __name__ == "__main__":
    main()
