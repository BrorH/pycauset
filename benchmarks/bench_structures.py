"""Structural-shortcut benchmark.

For operations with a properties-as-gospel shortcut, verify the shortcut returns the
same result as the general path on identical data, and measure the speedup.

Usage: python benchmarks/bench_structures.py
"""
from __future__ import annotations

import time

import numpy as np

import pycauset as pc

N = 800


def _timeit(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _make(n, structure):
    """Return (structured matrix, dense twin) with identical data.

    The structured matrix carries the property so the shortcut fires; the dense twin
    has the same data but no property, so it takes the general path.
    """
    rng = np.random.default_rng(0)
    if structure == "identity":
        data = np.eye(n)
        structured = pc.matrix(data)
        structured.properties["is_identity"] = True
        dense_twin = pc.matrix(data)
    elif structure == "zero":
        data = np.zeros((n, n))
        structured = pc.zeros((n, n), dtype="float64")
        dense_twin = pc.matrix(data)
    elif structure == "diagonal":
        data = np.diag(rng.standard_normal(n) + 2.0)
        structured = pc.matrix(data)
        structured.properties["is_diagonal"] = True
        dense_twin = pc.matrix(data)
    elif structure == "triangular":
        data = np.triu(rng.standard_normal((n, n)) + n * np.eye(n))
        structured = pc.matrix(data)
        structured.properties["is_upper_triangular"] = True
        dense_twin = pc.matrix(data)
    else:
        raise ValueError(structure)
    return structured, dense_twin


OPS = [
    ("matrix_rank", lambda m: pc.matrix_rank(m), ["identity", "diagonal", "triangular"]),
    ("trace", lambda m: pc.trace(m), ["identity", "diagonal"]),
    ("norm", lambda m: pc.norm(m), ["identity", "zero", "diagonal"]),
    ("determinant", lambda m: pc.determinant(m), ["identity", "diagonal", "triangular"]),
    ("matrix_power", lambda m: pc.matrix_power(m, 3), ["identity", "zero"]),
    ("invert", lambda m: pc.invert(m), ["identity"]),
]


def main():
    print("# Structural-shortcut benchmark")
    print(f"(n={N}; shortcut vs general path on identical data)")
    print()
    for name, fn, structures in OPS:
        print(f"## {name}")
        print("| structure | correct | shortcut time | general time | speedup |")
        print("|---|---|---|---|---|")
        for s in structures:
            structured, dense_twin = _make(N, s)
            t_fast = _timeit(lambda: fn(structured))
            try:
                t_dense = _timeit(lambda: fn(dense_twin), reps=1 if N >= 600 else 3)
                fast_result = fn(structured)
                dense_result = fn(dense_twin)
                correct = bool(np.allclose(np.asarray(fast_result), np.asarray(dense_result), atol=1e-4))
            except Exception:
                t_dense = float("nan")
                correct = False
            print(f"| {s} | {'yes' if correct else 'NO'} | {t_fast*1000:.2f}ms | {t_dense*1000:.2f}ms | {t_dense/t_fast:.0f}x |")
        print()


if __name__ == "__main__":
    main()
