"""PyCauset vs NumPy benchmark harness.

Produces a markdown-friendly table of timings for the headline operations.
Run from the repo root:  python benchmarks/bench.py

Method: `time.perf_counter`, best-of-N per operation. Dense float64 unless noted.
"""
from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np

import pycauset as pc


def _timeit(fn: Callable[[], None], repeats: int = 3) -> float:
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
    return f"{ms * 1000:.0f}µs"


def bench_matmul(sizes: list[int]) -> None:
    print("\n## Dense float64 matmul (C = A @ B)")
    print("| n | NumPy | PyCauset | ratio |")
    print("|---|---|---|---|")
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        a = pc.matrix(A)
        b = pc.matrix(B)
        t_np = _timeit(lambda: A @ B)
        t_pc = _timeit(lambda: a @ b)
        ratio = t_np / t_pc
        print(f"| {n} | {_fmt(t_np * 1000)} | {_fmt(t_pc * 1000)} | {ratio:.2f}x |")


def bench_factorizations(sizes: list[int]) -> None:
    print("\n## Dense float64 factorizations (LAPACK vs NumPy)")
    print("| op | n | NumPy | PyCauset | ratio |")
    print("|---|---|---|---|---|")
    for n in sizes:
        A = np.random.rand(n, n)
        A = A @ A.T + n * np.eye(n)  # SPD for cholesky/inverse
        a = pc.matrix(A)

        # invert() caches its result, so use a fresh matrix each iteration to
        # time the actual factorization rather than a cache hit.
        t_np = _timeit(lambda: np.linalg.inv(A))
        t_pc = _timeit(lambda: pc.invert(pc.matrix(A)))
        print(f"| inverse | {n} | {_fmt(t_np * 1000)} | {_fmt(t_pc * 1000)} | {t_np / t_pc:.2f}x |")

        t_np = _timeit(lambda: np.linalg.cholesky(A))
        t_pc = _timeit(lambda: pc.cholesky(a))
        print(f"| cholesky | {n} | {_fmt(t_np * 1000)} | {_fmt(t_pc * 1000)} | {t_np / t_pc:.2f}x |")

        b_np = np.random.rand(n, 1)
        b = pc.matrix(b_np)
        t_np = _timeit(lambda: np.linalg.solve(A, b_np))
        t_pc = _timeit(lambda: pc.solve(a, b))
        print(f"| solve | {n} | {_fmt(t_np * 1000)} | {_fmt(t_pc * 1000)} | {t_np / t_pc:.2f}x |")


def bench_bit_matrix(n: int) -> None:
    print("\n## Bit-packed boolean matrix (causal-set storage)")
    A = np.random.randint(0, 2, (n, n)).astype(bool)
    B = np.random.randint(0, 2, (n, n)).astype(bool)
    a = pc.matrix(A)
    b = pc.matrix(B)

    np_bytes = A.nbytes
    pc_bytes = n * n // 8
    print(f"| metric | NumPy bool | PyCauset bit | reduction |")
    print(f"|---|---|---|---|")
    print(f"| storage ({n}x{n}) | {np_bytes / 1e6:.1f} MB | {pc_bytes / 1e6:.1f} MB | {np_bytes / pc_bytes:.0f}x |")

    t_pc = _timeit(lambda: a @ b, repeats=2)
    print(f"| bit matmul ({n}x{n}) | - | {_fmt(t_pc * 1000)} | AVX-512 popcount |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matmul", type=int, nargs="*", default=[1000, 2000, 4000])
    ap.add_argument("--factor", type=int, nargs="*", default=[500, 1000, 2000])
    ap.add_argument("--bit", type=int, default=10000)
    args = ap.parse_args()

    print("# PyCauset benchmark results")
    print(f"(best-of-N, `time.perf_counter`; NumPy {np.__version__}; pycauset {pc.__version__})")
    bench_matmul(args.matmul)
    bench_factorizations(args.factor)
    bench_bit_matrix(args.bit)


if __name__ == "__main__":
    main()
