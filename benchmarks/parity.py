"""Canonical compute-parity benchmark (PyCauset vs NumPy, dense float64).

Reproduces with::

    python benchmarks/parity.py

For each op, reports the ratio ``numpy_time / pycauset_time``. A ratio >= 1.0
means PyCauset is faster; >= 0.90 means "at parity". Sizes are
chosen large enough that Python-dispatch overhead is amortized. Report-only (CI prints it).
The I/O rows (import, export, save, load + read) replace the former
benchmark_numpy_parity.py / benchmark_io_smoke.py; per-operation timing over every dtype and
structure is `python -m tests.atlas.timing`.
"""

from __future__ import annotations

import time

# pycauset first: it sets OPENBLAS_THREAD_TIMEOUT / OMP_WAIT_POLICY defaults before NumPy's own
# OpenBLAS loads. Otherwise NumPy's BLAS threads spin ~0.1 s after each NumPy call and slow the
# pycauset call timed next 3x (FINDINGS F-012; docs "Performance and threads").
import pycauset as pc  # noqa: I001

if pc._numpy_imported_first:  # (NumPy's timings would be the spinning threads, review 3.15 C-8)
    raise SystemExit("parity.py: NumPy was imported before pycauset; run it as a script")

# Time the computation, not the derived-value cache (a repeated det / sum / norm of the same
# object is served from it).
pc._native._set_derived_cache(False)

import numpy as np

N = 1024  # matrix ops
NV = 1_000_000  # vector/memory-bound ops


def best_time(fn, repeats=9):  # 3 is too noisy on a shared 4-vCPU machine
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
        # Force lazy results (LAZY.md) so the computation is timed; the NumPy export is timed
        # separately ("export" row) instead of inside every op.
        return pc.materialize(x) if getattr(x, "is_lazy", False) else x

    def run(name, np_fn, pc_fn):
        np_t = best_time(np_fn)
        # Force materialization so lazy PyCauset results are actually computed.
        pc_t = best_time(lambda: _materialize(pc_fn()))
        ratio = np_t / pc_t if pc_t > 0 else float("inf")
        verdict = "PASS" if ratio >= 0.90 else "FAIL"
        rows.append((name, ratio, verdict))
        print(f"{name:<14} {np_t:>12.4f} {pc_t:>12.4f} {ratio:>8.2f}x  {verdict}")

    # (The derived-value cache is off above, so det / inv recompute every call, as NumPy's.)
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="pycauset-parity-"))
    np_file, pc_file = tmp / "a.npy", tmp / "a.pycauset"

    run("matmul", lambda: A_np @ B_np, lambda: pc.matmul(A, B))
    run("solve", lambda: np.linalg.solve(spd_np, b_np), lambda: pc.solve(spd, b))
    run("inv", lambda: np.linalg.inv(spd_np), lambda: pc.inv(spd))
    run("dot", lambda: np.dot(v_np, v_np), lambda: pc.dot(v, v))
    run("add", lambda: A_np + B_np, lambda: A + B)
    run("multiply", lambda: A_np * B_np, lambda: A * B)
    run("determinant", lambda: np.linalg.det(spd_np), lambda: pc.det(spd))
    run("eigh", lambda: np.linalg.eigh(spd_np), lambda: pc.eigh(spd))
    run("export", lambda: A_np.copy(), lambda: np.asarray(A))  # to NumPy (a copy) vs ndarray.copy
    run("import", lambda: A_np.copy(), lambda: pc.matrix(A_np))  # from NumPy (a copy)
    # Saves without fsync on both sides would differ in what reaches the disk: pc.save fsyncs
    # (durable), so NumPy's side syncs too.
    def np_save():
        with open(np_file, "wb") as f:
            np.save(f, A_np)
            f.flush()
            import os

            os.fsync(f.fileno())

    run("save", np_save, lambda: pc.save(A, pc_file))
    # load + read every value (a load alone maps the file lazily: it measured the open).
    run("load+sum", lambda: np.load(np_file).sum(), lambda: pc.sum(pc.load(pc_file)))
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 62)
    fails = [r for r in rows if r[2] == "FAIL"]
    print(f"{len(rows) - len(fails)}/{len(rows)} ops at parity (>= 0.90x).")
    if fails:
        print("Below parity:", ", ".join(f"{n} ({r:.2f}x)" for n, r, _ in fails))


if __name__ == "__main__":
    main()
