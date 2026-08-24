"""PyCauset vs NumPy benchmark harness.

Measures the headline operations that have a direct NumPy equivalent, across a
range of sizes. Results are printed as markdown tables and saved to
`benchmarks/results.json` for `plot.py` to render graphs.

Usage:
    python benchmarks/bench.py            # default sizes
    python benchmarks/bench.py --large    # larger sizes (slower)

Method: `time.perf_counter`, best-of-N per operation (N scales down for large
sizes). Dense float64 unless noted.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import numpy as np

import pycauset as pc


def _timeit(fn: Callable[[], object], repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _fmt(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    if ms >= 1:
        return f"{ms:.1f}ms"
    return f"{ms * 1000:.0f}us"


def _spd(n: int) -> np.ndarray:
    """A well-conditioned symmetric positive-definite matrix."""
    a = np.random.rand(n, n)
    return a @ a.T + n * np.eye(n)


def _general(n: int) -> np.ndarray:
    """A well-conditioned general matrix."""
    return np.random.rand(n, n) + n * np.eye(n)


def _measure(op: str, sizes: list[int], np_fn: Callable[[int], object],
             pc_fn: Callable[[int], object], results: dict) -> None:
    """Measure one op across sizes; append to results and print a table."""
    print(f"\n## {op}")
    print("| n | NumPy | PyCauset | speedup |")
    print("|---|---|---|---|")
    out = {"sizes": [], "numpy_ms": [], "pycauset_ms": []}
    for n in sizes:
        # Fewer repeats for large, expensive sizes.
        repeats = 2 if n >= 4000 else (3 if n >= 1000 else 5)
        t_np = _timeit(lambda: np_fn(n), repeats)
        t_pc = _timeit(lambda: pc_fn(n), repeats)
        out["sizes"].append(n)
        out["numpy_ms"].append(round(t_np * 1000, 3))
        out["pycauset_ms"].append(round(t_pc * 1000, 3))
        speedup = t_np / t_pc
        print(f"| {n} | {_fmt(t_np * 1000)} | {_fmt(t_pc * 1000)} | {speedup:.2f}x |")
    results[op] = out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="use larger sizes (slower)")
    args = ap.parse_args()

    results: dict = {}
    if args.large:
        matmul_sizes = [256, 512, 1024, 2048, 4096, 8192]
        fact_sizes = [256, 512, 1024, 2048, 4096]
        svd_sizes = [256, 512, 1024, 2048]
        eig_sizes = [256, 512, 1024, 2048]
        elem_sizes = [1024, 2048, 4096, 8192]
        dot_sizes = [100_000, 1_000_000, 10_000_000]
    else:
        matmul_sizes = [256, 512, 1024, 2048, 4096]
        fact_sizes = [256, 512, 1024, 2048]
        svd_sizes = [256, 512, 1024]
        eig_sizes = [256, 512, 1024]
        elem_sizes = [512, 1024, 2048, 4096]
        dot_sizes = [100_000, 1_000_000]

    print("# PyCauset benchmark results")
    print(f"(best-of-N, time.perf_counter; NumPy {np.__version__}; pycauset {pc.__version__})")

    # matmul
    def np_matmul(n):
        A = _general(n); B = _general(n); return A @ B
    def pc_matmul(n):
        A = _general(n); B = _general(n); return pc.matrix(A) @ pc.matrix(B)
    _measure("matmul", matmul_sizes, np_matmul, pc_matmul, results)

    # inverse (fresh matrix per call to bypass invert()'s result cache)
    def np_inv(n):
        return np.linalg.inv(_general(n))
    def pc_inv(n):
        return pc.invert(pc.matrix(_general(n)))
    _measure("inverse", fact_sizes, np_inv, pc_inv, results)

    # solve
    def np_solve(n):
        A = _general(n); b = np.random.rand(n, 1); return np.linalg.solve(A, b)
    def pc_solve(n):
        A = _general(n); b = np.random.rand(n, 1); return pc.solve(pc.matrix(A), pc.matrix(b))
    _measure("solve", fact_sizes, np_solve, pc_solve, results)

    # cholesky
    def np_chol(n):
        return np.linalg.cholesky(_spd(n))
    def pc_chol(n):
        return pc.cholesky(pc.matrix(_spd(n)))
    _measure("cholesky", fact_sizes, np_chol, pc_chol, results)

    # svd (thin)
    def np_svd(n):
        return np.linalg.svd(_general(n), full_matrices=False)
    def pc_svd(n):
        return pc.svd(pc.matrix(_general(n)), full_matrices=False)
    _measure("svd", svd_sizes, np_svd, pc_svd, results)

    # eigh
    def np_eigh(n):
        return np.linalg.eigh(_spd(n))
    def pc_eigh(n):
        return pc.eigh(pc.matrix(_spd(n)))
    _measure("eigh", eig_sizes, np_eigh, pc_eigh, results)

    # eigvalsh
    def np_eigvalsh(n):
        return np.linalg.eigvalsh(_spd(n))
    def pc_eigvalsh(n):
        return pc.eigvalsh(pc.matrix(_spd(n)))
    _measure("eigvalsh", eig_sizes, np_eigvalsh, pc_eigvalsh, results)

    # elementwise add (construct + materialize, the honest end-to-end cost)
    def np_add(n):
        A = _general(n); B = _general(n); return A + B
    def pc_add(n):
        A = _general(n); B = _general(n); return np.asarray(pc.matrix(A) + pc.matrix(B))
    _measure("add", elem_sizes, np_add, pc_add, results)

    # dot (vectors pre-constructed to measure the operation, not construction)
    print("\n## dot")
    print("| n | NumPy | PyCauset | speedup |")
    print("|---|---|---|---|")
    dot_out = {"sizes": [], "numpy_ms": [], "pycauset_ms": []}
    for n in dot_sizes:
        a_np = np.random.rand(n); b_np = np.random.rand(n)
        a_v = pc.vector(a_np); b_v = pc.vector(b_np)
        t_np = _timeit(lambda: np.dot(a_np, b_np), 5)
        t_pc = _timeit(lambda: a_v.dot(b_v), 5)
        dot_out["sizes"].append(n)
        dot_out["numpy_ms"].append(round(t_np * 1000, 3))
        dot_out["pycauset_ms"].append(round(t_pc * 1000, 3))
        print(f"| {n} | {_fmt(t_np * 1000)} | {_fmt(t_pc * 1000)} | {t_np / t_pc:.2f}x |")
    results["dot"] = dot_out

    # Save results for plotting
    results["meta"] = {
        "numpy": np.__version__,
        "pycauset": pc.__version__,
        "large": args.large,
    }
    with open("benchmarks/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to benchmarks/results.json")


if __name__ == "__main__":
    main()
