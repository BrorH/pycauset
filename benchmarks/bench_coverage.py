"""Variant coverage benchmark: operation x dtype x structure.

For every operation with a NumPy equivalent, test correctness across all supported
dtypes and measure speed vs NumPy. Outputs:

1. A summary table (per op: dtypes passing, average speedup).
2. Per-op detail (dtype x result).

Usage: python benchmarks/bench_coverage.py
"""
from __future__ import annotations

import time

import numpy as np

import pycauset as pc

# dtype token -> numpy dtype (complex_float16 maps to complex64, matching the export guard)
DTYPES = {
    "bit": np.bool_,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "complex_float16": np.complex64,
    "complex_float32": np.complex64,
    "complex_float64": np.complex128,
}

N_CORRECT = 16   # matrix size for correctness checks (small = fast)
N_SPEED = 128    # matrix size for speed checks


def _pc_dtype(token: str) -> str:
    return token


def _make_np(n: int, dtype: str) -> np.ndarray:
    """A well-conditioned square matrix in the given dtype."""
    a = np.random.rand(n, n) + n * np.eye(n)
    return a.astype(DTYPES[dtype])


def _to_numpy(obj) -> np.ndarray:
    return np.asarray(obj)


def _timeit(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _check(got, want, dtype) -> tuple[bool, str]:
    """Compare a pycauset result (converted to numpy) against the numpy reference."""
    try:
        g = _to_numpy(got)
    except Exception as e:
        return False, f"export error: {type(e).__name__}"
    w = want
    try:
        if np.iscomplexobj(w) or np.iscomplexobj(g):
            return bool(np.allclose(g, w, atol=1e-4, rtol=1e-4)), ""
        if w.dtype == np.bool_ or w.dtype.kind in "iu":
            return bool(np.array_equal(g, w)), ""
        return bool(np.allclose(g, w, atol=1e-5, rtol=1e-5)), ""
    except Exception as e:
        return False, f"compare error: {type(e).__name__}"


# Each op: name, numpy function (A, B) -> result, pycauset function (a, b) -> result,
# dtypes it applies to, and whether it is a matrix-matrix op (B is a matrix).
OPS = [
    ("matmul", lambda A, B: A @ B, lambda a, b: a @ b, ["float16", "float32", "float64", "complex_float32", "complex_float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bit"]),
    ("add", lambda A, B: A + B, lambda a, b: a + b, ["float16", "float32", "float64", "complex_float32", "complex_float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bit"]),
    ("invert", lambda A, B: np.linalg.inv(A), lambda a, b: pc.invert(a), ["float32", "float64", "complex_float32", "complex_float64"]),
    ("solve", lambda A, B: np.linalg.solve(A, B), lambda a, b: pc.solve(a, b), ["float32", "float64", "complex_float32", "complex_float64"]),
    ("cholesky", lambda A, B: np.linalg.cholesky(A), lambda a, b: pc.cholesky(a), ["float32", "float64", "complex_float32", "complex_float64"]),
    ("svd", lambda A, B: np.linalg.svd(A, full_matrices=False)[1], lambda a, b: pc.svdvals(a), ["float32", "float64", "complex_float32", "complex_float64"]),
    ("svdvals", lambda A, B: np.linalg.svd(A, compute_uv=False), lambda a, b: pc.svdvals(a), ["float32", "float64", "complex_float32", "complex_float64"]),
    ("matrix_rank", lambda A, B: np.linalg.matrix_rank(A), lambda a, b: pc.matrix_rank(a), ["float32", "float64", "complex_float32", "complex_float64"]),
    ("matrix_power", lambda A, B: np.linalg.matrix_power(A, 3), lambda a, b: pc.matrix_power(a, 3), ["float32", "float64", "complex_float32", "complex_float64", "int32", "int64"]),
    ("norm", lambda A, B: np.linalg.norm(A, "fro"), lambda a, b: pc.norm(a), ["float16", "float32", "float64", "complex_float32", "complex_float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bit"]),
    ("determinant", lambda A, B: np.linalg.det(A), lambda a, b: pc.determinant(a), ["float32", "float64", "complex_float32", "complex_float64"]),
    ("eigvalsh", lambda A, B: np.linalg.eigvalsh(A), lambda a, b: pc.eigvalsh(a), ["float32", "float64", "complex_float32", "complex_float64"]),
    ("trace", lambda A, B: np.trace(A), lambda a, b: pc.trace(a), ["float16", "float32", "float64", "complex_float32", "complex_float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bit"]),
    ("outer", lambda A, B: np.outer(A[0, :], A[1, :]), lambda a, b: pc.outer(pc.vector(np.asarray(a)[0, :]), pc.vector(np.asarray(a)[1, :])), ["float32", "float64", "complex_float32", "complex_float64", "int32", "int64"]),
]


def _spd_np(n, dtype):
    """Symmetric positive-definite matrix (for cholesky/eigh)."""
    a = np.random.rand(n, n).astype(np.float64)
    a = a @ a.T + n * np.eye(n)
    return a.astype(DTYPES[dtype])


def main():
    summary = []
    print("# Variant coverage benchmark")
    print(f"(correctness at n={N_CORRECT}, speed at n={N_SPEED}, best-of-3; NumPy {np.__version__})")
    print()

    for name, np_fn, pc_fn, dtypes in OPS:
        results = []
        for dt in dtypes:
            # correctness
            if name in ("cholesky", "eigvalsh"):
                A = _spd_np(N_CORRECT, dt)
            else:
                A = _make_np(N_CORRECT, dt)
            B = _make_np(N_CORRECT, dt) if name not in ("matrix_rank", "norm", "determinant", "trace", "outer", "svdvals", "matrix_power", "cholesky", "eigvalsh") else None
            a = pc.matrix(A)
            b = pc.matrix(B) if B is not None else None
            try:
                want = np_fn(A, B)
            except Exception as e:
                want = e
            try:
                got = pc_fn(a, b)
                ok, msg = _check(got, want, dt)
            except Exception as e:
                ok, msg = False, f"{type(e).__name__}: {str(e)[:50]}"
            results.append((dt, ok, msg))

            # speed (dense float64 only, medium n)
        # speed for float64
        A = _make_np(N_SPEED, "float64")
        B = _make_np(N_SPEED, "float64")
        a = pc.matrix(A); b = pc.matrix(B)
        try:
            t_np = _timeit(lambda: np_fn(A, B))
            t_pc = _timeit(lambda: pc_fn(a, b))
            speedup = t_np / t_pc
        except Exception:
            speedup = float("nan")

        passing = sum(1 for _, ok, _ in results if ok)
        print(f"## {name}")
        print(f"dtypes passing: {passing}/{len(dtypes)}; float64 speedup: {speedup:.2f}x")
        for dt, ok, msg in results:
            mark = "OK" if ok else f"FAIL ({msg})"
            print(f"  {dt:18s} {mark}")
        print()
        summary.append((name, passing, len(dtypes), speedup))

    print("\n## Summary")
    print("| op | dtypes passing | float64 speedup |")
    print("|---|---|---|")
    for name, p, t, s in summary:
        print(f"| {name} | {p}/{t} | {s:.2f}x |")


if __name__ == "__main__":
    main()
