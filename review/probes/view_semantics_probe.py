"""Which (class, op) pairs return an O(1) view today, are the values right, and does
mutating the source afterwards leak into the result?  Input for review/features/LAZY.md.

Run: python review/probes/view_semantics_probe.py   (from the repo root)
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "python")
import pycauset as pc  # noqa: E402

N = 6
rng = np.random.default_rng(0)


def ex(x):
    return np.asarray(pc.to_numpy(x, allow_huge=True))


def dense(dt):
    if dt == "bool":
        a = rng.integers(0, 2, (N, N)).astype(bool)
    elif "complex" in dt:
        a = (rng.integers(-4, 5, (N, N)) + 1j * rng.integers(-4, 5, (N, N))).astype(dt)
    elif dt.startswith("uint"):
        a = rng.integers(0, 9, (N, N)).astype(dt)
    else:
        a = rng.integers(-4, 5, (N, N)).astype(dt)
    return pc.matrix(a), (0, 1)


def structured():
    out = {}
    t = pc.TriangularFloatMatrix(N)
    for i in range(N):
        for j in range(i + 1, N):
            t.set(i, j, float(rng.integers(-4, 5)))
    out["TriangularFloatMatrix"] = (t, (0, 1))
    ti = pc.TriangularIntegerMatrix(N) if hasattr(pc, "TriangularIntegerMatrix") else pc._native.TriangularIntegerMatrix(N)
    for i in range(N):
        for j in range(i + 1, N):
            ti.set(i, j, int(rng.integers(-4, 5)))
    out["TriangularIntegerMatrix"] = (ti, (0, 1))
    out["TriangularBitMatrix(causal)"] = (pc.causal_matrix(N, populate=True, seed=1), (0, 1))
    d = pc._native.DiagonalMatrix(N)
    for i in range(N):
        d.set(i, i, float(i + 1))
    out["DiagonalMatrix"] = (d, (0, 0))
    s = pc._native.SymmetricMatrix(N)
    a = pc._native.AntiSymmetricMatrix(N)
    for i in range(N):
        for j in range(i, N):
            s.set(i, j, float(rng.integers(-4, 5)))
            if j > i:
                a.set(i, j, float(rng.integers(-4, 5)))
    out["SymmetricMatrix"] = (s, (0, 1))
    out["AntiSymmetricMatrix"] = (a, (0, 1))
    out["IdentityMatrix"] = (pc._native.IdentityMatrix(N), (0, 0))
    return out


OPS = {
    "2*A": (lambda A: 2 * A, lambda a: 2 * a),
    "A*2.5": (lambda A: A * 2.5, lambda a: a * 2.5),
    "-A": (lambda A: -A, lambda a: -(a.astype(np.int64) if a.dtype == bool else a)),
    "A/2": (lambda A: A / 2, lambda a: a / 2),
    "A.T": (lambda A: A.T, lambda a: a.T),
    "A.conj()": (lambda A: A.conj(), lambda a: np.conj(a)),
    "A.H": (lambda A: A.H, lambda a: np.conj(a).T),
}


def is_view(r):
    try:
        return r.scalar != 1 or r.is_transposed() or r.is_conjugated()
    except Exception:
        return False


def main():
    objs = {}
    for dt in ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
               "float16", "float32", "float64", "complex64", "complex128"]:
        m, pos = dense(dt)
        objs[f"{type(m).__name__}({dt})"] = (m, pos)
    objs.update(structured())

    print(f"{'class':34s} " + " ".join(f"{k:>9s}" for k in OPS))
    for name, (A, pos) in objs.items():
        cells = []
        for opname, (pf, nf) in OPS.items():
            try:
                before = ex(A).copy()
                R = pf(A)
                want = nf(before)
                got = ex(R)
                ok = np.allclose(got.astype(np.complex128), np.asarray(want).astype(np.complex128), atol=1e-2)
                view = is_view(R)
                # mutate the source; does R change?
                leak = "?"
                try:
                    snap = got.copy()
                    old = A.get(*pos) if hasattr(A, "get") else None
                    A.set(*pos, 1 if before.dtype == bool else 3)
                    leak = "LEAK" if not np.array_equal(ex(R), snap) else ""
                    A.set(*pos, old)
                except Exception:
                    leak = "ro"
                cell = ("v" if view else "c") + ("" if ok else "!WRONG") + (f"+{leak}" if leak else "")
            except Exception as e:  # noqa: BLE001
                cell = "ERR:" + type(e).__name__
            cells.append(cell)
        print(f"{name:34s} " + " ".join(f"{c:>9s}" for c in cells))
    print("\nlegend: v = O(1) view (scalar/transpose/conjugate flag), c = new storage, "
          "!WRONG = values differ from NumPy, +LEAK = writing to A afterwards changed the result, "
          "ro = source not writable, ERR = op raised")


if __name__ == "__main__":
    main()
