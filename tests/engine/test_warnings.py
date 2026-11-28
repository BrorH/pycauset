"""PyCauset's warnings: what triggers each one, its category, and that it never changes a result.

Policy (CLAUDE.md): guesses may warn, never decide. Each test also checks the value, so a
warning path that altered the answer would fail here. Moved from the old suite
(test_float_underpromotion_warning, test_integer_matmul_overflow[_risk_warning]).
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import pycauset as pc


def _caught(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    return out, w


def _of(w, category, text):
    return [x for x in w if issubclass(x.category, category) and text in str(x.message)]


def test_float_underpromotion_warns_every_call_and_returns_float32():
    a = pc.matrix(np.diag([1.0, 1.0]).astype(np.float32))
    b = pc.matrix(np.diag([2.0, 3.0]))
    (c1, c2), w = _caught(lambda: (a @ b, a @ b))
    # Python's warnings filter owns deduplication: under "always" every call warns.
    assert len(_of(w, pc.PyCausetDTypeWarning, "float32 with float64 gives float32")) == 2
    for c in (c1, c2):
        assert c.dtype == "float32"  # F-034: within-float under-promotion
        np.testing.assert_array_equal(np.asarray(c), np.diag([2.0, 3.0]).astype(np.float32))


def test_bit_arithmetic_warns_int32():
    a = pc.matrix(np.ones((2, 2), dtype=bool))
    out, w = _caught(lambda: a + a)
    assert _of(w, pc.PyCausetDTypeWarning, "bit operands promote to int32")
    assert out.dtype == "int32"
    np.testing.assert_array_equal(np.asarray(out), np.full((2, 2), 2))


def test_integer_matmul_overflow_risk_is_advisory():
    n, v = 256, 46340  # v * v < 2^31 - 1, but n * v * v does not fit: the estimate warns
    x = np.zeros((n, n), dtype=np.int32)
    x[0, 0] = v
    a = pc.matrix(x)
    c, w = _caught(lambda: a @ a)
    hits = _of(w, pc.PyCausetOverflowRiskWarning, "may overflow the int32 output")
    assert hits and "advisory estimate" in str(hits[0].message)
    assert c[0, 0] == v * v  # the warning changed nothing


def test_integer_matmul_overflow_raises():
    x = np.zeros((2, 2), dtype=np.int32)
    x[0, 0] = 2**31 - 1
    y = np.zeros((2, 2), dtype=np.int32)
    y[0, 0] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(OverflowError):
            pc.matrix(x) @ pc.matrix(y)  # F-055


def test_numpy_imported_first_warns_once_about_two_blas_pools():
    code = "import numpy, warnings; warnings.simplefilter('always'); import pycauset"
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_THREAD_TIMEOUT"}
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(pc.__file__)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stderr.count("PyCausetPerformanceWarning") == 1


@pytest.mark.parametrize("make", [
    lambda: pc.causal_matrix(np.zeros((3, 3), dtype=bool)),
    lambda: pc.TriangularMatrix(3, dtype="float64", has_diagonal=True),
])
def test_writing_outside_the_stored_triangle_raises(make):
    m = make()
    with pytest.raises(ValueError):
        m[2, 0] = 1


def test_the_engine_never_prints():
    """Information reaches the user as a warning, an exception, the IO trace or debug_mode,
    never as stdout / stderr from C++ (streamed matmuls and file-backed allocations used to
    print). The parked CUDA hooks are the one exception (experimental/cuda/README.md)."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    allowed = {"AutoSolver.cpp", "ComputeContext.cpp"}
    offenders = [
        f"{p.relative_to(root)}:{i}"
        for d in ("src", "include") for p in sorted((root / d).rglob("*.[ch]pp")) if p.name not in allowed
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
        if re.search(r"std::(cout|cerr)|\bprintf\(", line) and not line.lstrip().startswith("//")
    ]
    assert not offenders, offenders


@pytest.mark.parametrize("da, db", [("float16", "float32"), ("float16", "float64"),
                                    ("complex64", "complex128"), ("complex64", "float64")])
def test_every_narrower_float_result_warns(da, db):
    """3.3 C-6: only float32 with float64 warned; float16 and complex mixes were silent."""
    a, b = pc.matrix(np.ones((2, 2), dtype=da)), pc.matrix(np.ones((2, 2), dtype=db))
    for op in (lambda: a + b, lambda: a @ b):
        _, w = _caught(op)
        assert _of(w, pc.PyCausetDTypeWarning, "NumPy would widen")


def test_same_width_and_widening_results_do_not_warn():
    a, b = pc.matrix(np.ones((2, 2), dtype="float32")), pc.matrix(np.ones((2, 2), dtype="complex64"))
    _, w = _caught(lambda: a + b)
    assert not _of(w, pc.PyCausetDTypeWarning, "NumPy would widen")


def test_a_wider_numpy_scalar_warns():
    """DType System §9: a NumPy scalar never widens a float matrix; say so."""
    m = pc.matrix(np.ones((2, 2), dtype=np.float32))
    out, w = _caught(lambda: m + np.float64(1))
    assert out.dtype == "float32" and _of(w, pc.PyCausetDTypeWarning, "NumPy would widen")
    _, w = _caught(lambda: m + 1.0)  # Python floats are weak: nothing to say
    assert not _of(w, pc.PyCausetDTypeWarning, "NumPy would widen")
