"""Probe: do ops give the same answers when operands are forced to disk?

Sets the memory threshold to a few KB so every matrix is file-backed, then compares
results against NumPy and reports where each operand/result actually lives.

Run:  python review/probes/out_of_core_probe.py
"""
from __future__ import annotations

import tempfile

import numpy as np

import pycauset as pc


def arr(x):
    """Export without the out-of-core guard (probe matrices are small)."""
    if isinstance(x, (int, float, complex, np.generic)):
        return np.asarray(x)
    return pc.to_numpy(x, allow_huge=True)


def where(x):
    try:
        bf = x.get_backing_file()
    except Exception:  # noqa: BLE001
        return "?"
    return "RAM" if bf in (None, "", ":memory:") else "disk"


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="pc_ooc_")
    pc.set_backing_dir(tmp)
    rng = np.random.default_rng(1)
    n = 300
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    B = rng.standard_normal((n, n))

    for label, thresh in (("in-RAM", None), ("forced-disk", 4096)):
        pc.set_memory_threshold(thresh if thresh is not None else 8 * 1024**3)
        pA, pB = pc.matrix(A), pc.matrix(B)
        print(f"--- {label}: operands live in {where(pA)}/{where(pB)}")
        cases = {
            "A@B": (lambda: pA @ pB, A @ B),
            "A+B": (lambda: pA + pB, A + B),
            "A-B": (lambda: pA - pB, A - B),
            "A*B": (lambda: pA * pB, A * B),
            "2*A": (lambda: 2.0 * pA, 2.0 * A),
            "A.T@B": (lambda: pA.T @ pB, A.T @ B),
            "inv(A)": (lambda: pc.invert(pA), np.linalg.inv(A)),
            "solve": (lambda: pc.solve(pA, pB), np.linalg.solve(A, B)),
            "det(A/n)": (lambda: pc.determinant(pc.matrix(A / n)), np.linalg.det(A / n)),
            "sum": (lambda: pc.sum(pA), A.sum()),
        }
        for name, (f, ref) in cases.items():
            try:
                r = f()
                if hasattr(r, "eval"):
                    r = r.eval()
                ok = np.allclose(arr(r), ref, rtol=1e-7, atol=1e-7)
                print(f"  {name:<9} ok={ok!s:<5} result in {where(r)}")
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<9} ERROR {type(e).__name__}: {str(e)[:120]}")

    # Causal (bit) matrices forced to disk
    pc.set_memory_threshold(4096)
    C = pc.causal_matrix(500, populate=True)
    Cn = arr(C).astype(np.int64)
    for name, f, ref in (
        ("C@C", lambda: C @ C, Cn @ Cn),
        ("C+C", lambda: C + C, Cn + Cn),
        ("C.T", lambda: C.T, Cn.T),
    ):
        try:
            r = f()
            ok = np.array_equal(arr(r).astype(np.int64), ref)
            print(f"  causal {name:<5} ok={ok!s:<5} type={type(r).__name__} in {where(r)}")
        except Exception as e:  # noqa: BLE001
            print(f"  causal {name:<5} ERROR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
