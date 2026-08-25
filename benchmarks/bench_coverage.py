"""Variant coverage benchmark: operation x dtype (tri-state classifier).

Classifies each (op, dtype) cell as one of:

- ok        : matches NumPy.
- by-design : raises a documented error (overflow / singular / non-SPD / not-implemented).
- WRONG     : silent wrong answer, or an unexpected error, or a broken NumPy export.

Usage: python benchmarks/bench_coverage.py
"""
from __future__ import annotations

import time

import numpy as np

import pycauset as pc

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
    "complex_float32": np.complex64,
    "complex_float64": np.complex128,
}

N_CORRECT = 16
N_SPEED = 256

# Documented "error by design" exception types: these count as correct behavior.
_BY_DESIGN = (OverflowError, np.linalg.LinAlgError, ValueError, NotImplementedError)


def _make_np(n, dtype):
    # Small integer values so integer matmul does not overflow, and a well-conditioned
    # float matrix for factorizations.
    nd = DTYPES[dtype]
    if nd == np.bool_:
        return np.random.randint(0, 2, (n, n)).astype(nd)
    if np.issubdtype(nd, np.integer):
        return np.random.randint(0, 3, (n, n)).astype(nd)
    a = np.random.rand(n, n) + n * np.eye(n)
    return a.astype(nd)


def _spd_np(n, dtype):
    nd = DTYPES[dtype]
    a = np.random.rand(n, n)
    a = a @ a.T + n * np.eye(n)
    return a.astype(nd)


def _promote_for_reference(A, dtype):
    """NumPy reference that mirrors PyCauset's promotion rules for bit/int."""
    nd = DTYPES[dtype]
    if nd == np.bool_:
        return A.astype(np.int32)  # PyCauset promotes bit matmul to int32
    return A


def _tol_for(w):
    if getattr(w, "dtype", None) == np.float16:
        return 1e-2  # float16 has ~3 decimal digits of precision
    if getattr(w, "dtype", None) in (np.float32, np.complex64):
        return 1e-4
    return 1e-5


def _classify(pc_fn, np_want):
    try:
        got = pc_fn()
    except _BY_DESIGN as e:
        return "by-design", type(e).__name__
    except Exception as e:
        return "WRONG", f"unexpected {type(e).__name__}: {str(e)[:40]}"
    try:
        g = np.asarray(got)
    except Exception as e:
        return "WRONG", f"export {type(e).__name__}"
    w = np_want
    tol = _tol_for(w)
    try:
        if np.iscomplexobj(w) or np.iscomplexobj(g):
            ok = bool(np.allclose(g, w, atol=tol, rtol=tol))
        elif w.dtype == np.bool_ or w.dtype.kind in "iu":
            ok = bool(np.array_equal(g, w))
        else:
            ok = bool(np.allclose(g, w, atol=tol, rtol=tol))
    except Exception as e:
        return "WRONG", f"compare {type(e).__name__}"
    return ("ok", "") if ok else ("WRONG", f"value mismatch (got {g.ravel()[:3]} want {w.ravel()[:3]})")


def _timeit(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# op -> (numpy fn(A,B), pycauset fn(a,b), dtypes, needs_second_matrix)
OPS = [
    ("matmul", lambda A, B: A @ B, lambda a, b: a @ b, ["bit", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex_float32", "complex_float64"], True),
    ("add", lambda A, B: A + B, lambda a, b: a + b, ["bit", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex_float32", "complex_float64"], True),
    ("invert", lambda A, B: np.linalg.inv(A), lambda a, b: pc.invert(a), ["float32", "float64", "complex_float32", "complex_float64"], False),
    ("solve", lambda A, B: np.linalg.solve(A, B), lambda a, b: pc.solve(a, b), ["float32", "float64", "complex_float32", "complex_float64"], True),
    ("cholesky", lambda A, B: np.linalg.cholesky(A), lambda a, b: pc.cholesky(a), ["float32", "float64", "complex_float32", "complex_float64"], False),
    ("svdvals", lambda A, B: np.linalg.svd(A, compute_uv=False), lambda a, b: pc.svdvals(a), ["float32", "float64", "complex_float32", "complex_float64"], False),
    ("matrix_rank", lambda A, B: np.linalg.matrix_rank(A), lambda a, b: pc.matrix_rank(a), ["float32", "float64", "complex_float32", "complex_float64"], False),
    ("matrix_power", lambda A, B: np.linalg.matrix_power(A, 3), lambda a, b: pc.matrix_power(a, 3), ["float32", "float64", "complex_float32", "complex_float64", "int32", "int64"], False),
    ("norm", lambda A, B: np.linalg.norm(A, "fro"), lambda a, b: pc.norm(a), ["bit", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex_float32", "complex_float64"], False),
    ("determinant", lambda A, B: np.linalg.det(A), lambda a, b: pc.determinant(a), ["float32", "float64", "complex_float32", "complex_float64"], False),
    ("eigvalsh", lambda A, B: np.linalg.eigvalsh(A), lambda a, b: pc.eigvalsh(a), ["float32", "float64", "complex_float32", "complex_float64"], False),
    ("trace", lambda A, B: np.trace(A), lambda a, b: pc.trace(a), ["bit", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex_float32", "complex_float64"], False),
    ("outer", lambda A, B: np.outer(A[0, :], A[1, :]), lambda a, b: pc.outer(pc.vector(np.asarray(a)[0, :]), pc.vector(np.asarray(a)[1, :])), ["float32", "float64", "complex_float32", "complex_float64", "int32", "int64"], False),
]


def main():
    print("# Variant coverage benchmark (op x dtype)")
    print(f"(correctness n={N_CORRECT}; speed n={N_SPEED} best-of-3; NumPy {np.__version__})")
    print()
    rows = []
    for name, np_fn, pc_fn, dtypes, needs_B in OPS:
        statuses = {}
        for dt in dtypes:
            if name in ("cholesky", "eigvalsh"):
                A = _spd_np(N_CORRECT, dt)
            else:
                A = _make_np(N_CORRECT, dt)
            A_ref = _promote_for_reference(A, dt)
            B = _make_np(N_CORRECT, dt) if needs_B else None
            B_ref = _promote_for_reference(B, dt) if B is not None else None
            a = pc.matrix(A)
            b = pc.matrix(B) if B is not None else None
            want = np_fn(A_ref, B_ref)
            status, note = _classify(lambda: pc_fn(a, b), want)
            statuses[dt] = (status, note)

        # speed (float64)
        A = _make_np(N_SPEED, "float64")
        B = _make_np(N_SPEED, "float64") if needs_B else None
        a = pc.matrix(A)
        b = pc.matrix(B) if B is not None else None
        try:
            t_np = _timeit(lambda: np_fn(A, B))
            t_pc = _timeit(lambda: pc_fn(pc.matrix(A), pc.matrix(B)) if needs_B else pc_fn(pc.matrix(A), None))
            speedup = t_np / t_pc
        except Exception:
            speedup = float("nan")

        n_ok = sum(1 for s, _ in statuses.values() if s == "ok")
        n_bd = sum(1 for s, _ in statuses.values() if s == "by-design")
        n_wrong = sum(1 for s, _ in statuses.values() if s == "WRONG")
        rows.append((name, n_ok, n_bd, n_wrong, len(dtypes), speedup))

        print(f"## {name}: ok={n_ok} by-design={n_bd} WRONG={n_wrong} speedup(f64)={speedup:.2f}x")
        for dt, (s, note) in statuses.items():
            if s == "ok":
                continue
            print(f"  {dt:16s} {s:10s} {note}")
        print()

    print("\n## Summary")
    print("| op | ok | by-design | WRONG | speedup(f64) |")
    print("|---|---|---|---|---|")
    for name, n_ok, n_bd, n_wrong, total, speedup in rows:
        flag = "" if n_wrong == 0 else "  <-- investigate"
        print(f"| {name} | {n_ok}/{total} | {n_bd} | {n_wrong} | {speedup:.2f}x |{flag}")

    print()
    bench_by_design()


def bench_by_design():
    """Verify documented errors are raised (and would be classified 'by-design')."""
    print("## Error-by-design checks")
    cases = [
        ("int8 matmul overflow", OverflowError,
         lambda: pc.matrix(np.full((16, 16), 100, dtype=np.int8)) @ pc.matrix(np.full((16, 16), 100, dtype=np.int8))),
        ("singular solve", np.linalg.LinAlgError,
         lambda: pc.solve(pc.matrix(np.array([[1.0, 2.0], [2.0, 4.0]])), pc.matrix(np.ones((2, 1))))),
        ("zero-marked solve", ValueError,
         lambda: pc.solve(pc.zeros((4, 4), dtype="float64"), pc.matrix(np.ones((4, 1))))),
        ("non-SPD cholesky", np.linalg.LinAlgError,
         lambda: pc.cholesky(pc.matrix(np.array([[1.0, 2.0], [2.0, 1.0]])))),
        ("non-triangular solve_triangular", ValueError,
         lambda: pc.solve_triangular(pc.matrix(np.eye(2)), pc.vector([1.0, 2.0]))),
    ]
    for label, expected, fn in cases:
        try:
            fn()
            print(f"  {label:32s} NO ERROR  <-- investigate")
        except expected as e:
            print(f"  {label:32s} {type(e).__name__} (by-design)")
        except Exception as e:
            print(f"  {label:32s} {type(e).__name__}  <-- unexpected")


if __name__ == "__main__":
    main()
