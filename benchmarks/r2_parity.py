"""R2_PERF — canonical compute-parity benchmark (PyCauset vs NumPy, dense float64).

Reproduces with::

    python benchmarks/r2_parity.py

For each op, reports the ratio ``numpy_time / pycauset_time``. A ratio >= 1.0
means PyCauset is faster; >= 0.90 means "at parity" (the R2_PERF bar). Sizes are
chosen large enough that Python-dispatch overhead is amortized. This is the single
canonical gate — it supersedes the older ad-hoc benchmark scripts.
"""

from __future__ import annotations

import time

import numpy as np

import pycauset as pc

N = 1024  # matrix ops
NV = 1_000_000  # vector/memory-bound ops


def best_time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    A_np = rng.standard_normal((N, N))
    B_np = rng.standard_normal((N, N))
    b_np = rng.standard_normal(N)
    v_np = rng.standard_normal(NV)

    A = pc.matrix(A_np)
    B = pc.matrix(B_np)
    b = pc.vector(b_np)
    v = pc.vector(v_np)

    # SPD matrix for solve/invert/eigh/det; scaled so det does not overflow.
    spd_np = (A_np @ A_np.T) / N + np.eye(N)
    spd = pc.matrix(spd_np)

    print(f"{'op':<14} {'numpy(s)':>12} {'pycauset(s)':>12} {'ratio':>8}  verdict")
    print("-" * 62)
    rows = []

    def _materialize(x):
        try:
            return np.asarray(x)
        except Exception:
            return x

    def run(name, np_fn, pc_fn):
        np_t = best_time(np_fn)
        # Force materialization so lazy PyCauset results are actually computed.
        pc_t = best_time(lambda: _materialize(pc_fn()))
        ratio = np_t / pc_t if pc_t > 0 else float("inf")
        verdict = "PASS" if ratio >= 0.90 else "FAIL"
        rows.append((name, ratio, verdict))
        print(f"{name:<14} {np_t:>12.4f} {pc_t:>12.4f} {ratio:>8.2f}x  {verdict}")

    # invert() and determinant() memoize their results (a real PyCauset feature:
    # the inverse lives in `_cached_inverse`, the determinant in the derived-
    # property store). NumPy recomputes every call, so a warm-cache comparison
    # would measure a cache hit against a fresh factorization — not parity. Clear
    # the derived caches before each timed call so both sides recompute.
    def _clear_derived_caches(m):
        try:
            from pycauset._internal.properties import _invalidate_cached_derived

            _invalidate_cached_derived(m)
        except Exception:
            if hasattr(m, "_cached_inverse"):
                del m._cached_inverse

    def _pc_invert():
        _clear_derived_caches(spd)
        return pc.invert(spd)

    def _pc_determinant():
        _clear_derived_caches(spd)
        return pc.determinant(spd)

    run("matmul", lambda: A_np @ B_np, lambda: pc.matmul(A, B))
    run("solve", lambda: np.linalg.solve(spd_np, b_np), lambda: pc.solve(spd, b))
    run("invert", lambda: np.linalg.inv(spd_np), _pc_invert)
    run("dot", lambda: np.dot(v_np, v_np), lambda: pc.dot(v, v))
    run("add", lambda: A_np + B_np, lambda: A + B)
    run("multiply", lambda: A_np * B_np, lambda: A * B)
    run("determinant", lambda: np.linalg.det(spd_np), _pc_determinant)
    run("eigh", lambda: np.linalg.eigh(spd_np), lambda: pc.eigh(spd))

    print("-" * 62)
    fails = [r for r in rows if r[2] == "FAIL"]
    print(f"{len(rows) - len(fails)}/{len(rows)} ops at parity (>= 0.90x).")
    if fails:
        print("Below parity:", ", ".join(f"{n} ({r:.2f}x)" for n, r, _ in fails))


if __name__ == "__main__":
    main()
