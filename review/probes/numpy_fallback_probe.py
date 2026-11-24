"""Probe: which pycauset ops (a) match NumPy and (b) secretly call numpy.linalg.

Every numpy.linalg function is wrapped to record calls. A pycauset op that records
a numpy.linalg call is materializing the matrix into RAM via NumPy, i.e. it cannot
work out-of-core no matter what the storage layer does.

Run:  python review/probes/numpy_fallback_probe.py
"""
from __future__ import annotations

import functools
import traceback

import numpy as np

import pycauset as pc

CALLS: list[str] = []


def _instrument() -> None:
    for name in dir(np.linalg):
        fn = getattr(np.linalg, name)
        if callable(fn) and not name.startswith("_") and not isinstance(fn, type):
            @functools.wraps(fn)
            def wrapper(*a, __fn=fn, __name=name, **k):
                CALLS.append(__name)
                return __fn(*a, **k)
            setattr(np.linalg, name, wrapper)


def to_np(x):
    if isinstance(x, tuple):
        return tuple(to_np(v) for v in x)
    if isinstance(x, (int, float, complex, np.generic, np.ndarray)):
        return np.asarray(x)
    return np.asarray(x)


def close(a, b, tol=1e-8):
    if isinstance(a, tuple):
        return all(close(x, y, tol) for x, y in zip(a, b))
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    return bool(np.allclose(a, b, rtol=tol, atol=tol))


def main() -> None:
    rng = np.random.default_rng(0)
    n = 40
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    S = A + A.T
    B = rng.standard_normal((n, 3))
    v = rng.standard_normal(n)
    pA, pS, pB, pv = pc.matrix(A), pc.matrix(S), pc.matrix(B), pc.vector(v)

    # (name, pycauset thunk, numpy reference thunk or None, compare-sorted-eigs?)
    cases = [
        ("matmul", lambda: pc.matmul(pA, pA), lambda: A @ A),
        ("dot(vec,vec)", lambda: pc.dot(pv, pv), lambda: v @ v),
        ("invert", lambda: pc.invert(pA), lambda: np.linalg.inv(A)),
        ("solve", lambda: pc.solve(pA, pB), lambda: np.linalg.solve(A, B)),
        ("determinant", lambda: pc.determinant(pA), lambda: np.linalg.det(A)),
        ("slogdet", lambda: pc.slogdet(pA), lambda: np.linalg.slogdet(A)),
        ("trace", lambda: pc.trace(pA), lambda: np.trace(A)),
        ("norm(fro)", lambda: pc.norm(pA), lambda: np.linalg.norm(A)),
        ("cond", lambda: pc.cond(pA), lambda: np.linalg.cond(A)),
        ("eigvalsh", lambda: np.sort(to_np(pc.eigvalsh(pS))), lambda: np.sort(np.linalg.eigvalsh(S))),
        ("eigh(vals)", lambda: np.sort(to_np(pc.eigh(pS)[0])), lambda: np.sort(np.linalg.eigh(S)[0])),
        ("eigvals", lambda: np.sort_complex(to_np(pc.eigvals(pA))), lambda: np.sort_complex(np.linalg.eigvals(A))),
        ("svdvals", lambda: pc.svdvals(pA), lambda: np.linalg.svd(A, compute_uv=False)),
        ("matrix_rank", lambda: pc.matrix_rank(pA), lambda: np.linalg.matrix_rank(A)),
        ("matrix_power(3)", lambda: pc.matrix_power(pA, 3), lambda: np.linalg.matrix_power(A, 3)),
        ("pinv", lambda: pc.pinv(pA), lambda: np.linalg.pinv(A)),
        ("lstsq(x)", lambda: to_np(pc.lstsq(pA, pB))[0] if isinstance(pc.lstsq(pA, pB), tuple) else pc.lstsq(pA, pB), lambda: np.linalg.lstsq(A, B, rcond=None)[0]),
        ("cholesky", lambda: pc.cholesky(pc.matrix(S + 2 * n * np.eye(n))), lambda: np.linalg.cholesky(S + 2 * n * np.eye(n))),
        ("qr(R abs)", lambda: np.abs(to_np(pc.qr(pA)[1])), lambda: np.abs(np.linalg.qr(A)[1])),
        ("outer", lambda: pc.outer(pv, pv), lambda: np.outer(v, v)),
        ("sum", lambda: pc.sum(pA), lambda: A.sum()),
        ("A+A", lambda: pA + pA, lambda: A + A),
        ("A*2.5", lambda: pA * 2.5, lambda: A * 2.5),
        ("A.T", lambda: pA.T, lambda: A.T),
    ]

    _instrument()
    width = max(len(c[0]) for c in cases)
    print(f"{'op':<{width}}  match   numpy.linalg calls")
    for name, f, ref in cases:
        CALLS.clear()
        try:
            got = f()
            used = sorted(set(CALLS))
            CALLS.clear()
            exp = ref()
            ok = close(to_np(got), exp)
            print(f"{name:<{width}}  {str(ok):<6}  {', '.join(used) or '-'}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:<{width}}  ERROR   {type(e).__name__}: {e}".splitlines()[0][:160])


if __name__ == "__main__":
    main()
