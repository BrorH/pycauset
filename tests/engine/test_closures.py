"""C ± Cᵀ closure views over causal bits (LAZY.md L2b).

`C + C.T` and `C - C.T` (C a causal matrix) are computed on read from C's bits. The rule
(LAZY.md section 2): indistinguishable from the eager result except for speed and memory.
Every consumer is compared with the same expression evaluated eagerly (`_set_lazy(False)`)
and with NumPy.
"""
from __future__ import annotations

import contextlib

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

native = pc._native
N = 23


@contextlib.contextmanager
def eager():
    native._set_lazy(False)
    try:
        yield
    finally:
        native._set_lazy(True)


EXPRS = {
    "C+C.T": (lambda c: c + c.T, lambda a: a + a.T),
    "C-C.T": (lambda c: c - c.T, lambda a: a - a.T),
    "C.T-C": (lambda c: c.T - c, lambda a: a.T - a),
    "C.T+C": (lambda c: c.T + c, lambda a: a.T + a),
}

CONSUMERS = {
    "to_numpy": lambda m: to_np(m),
    "getitem": lambda m: np.array([[m[i, j] for j in range(N)] for i in range(N)]),
    "T": lambda m: to_np(m.T),
    "neg": lambda m: to_np(-m),
    "times3": lambda m: to_np(3 * m),
    "plus1": lambda m: to_np(m + 1),
    "plus_dense": lambda m: to_np(m + pc.matrix(np.ones((N, N), dtype=np.int32))),
    "matvec_float": lambda m: to_np(m @ pc.vector(np.linspace(-1, 1, N))),
    "matvec_int": lambda m: to_np(m @ pc.vector(np.arange(N, dtype=np.int32))),
    "matmul_float": lambda m: to_np(m @ pc.matrix(np.eye(N) * 2.0)),
    "sum": lambda m: complex(pc.sum(m)).real,
    "trace": lambda m: pc.trace(m),
    "materialize": lambda m: to_np(pc.materialize(m)),
}


@pytest.fixture(scope="module")
def causal():
    c = pc.causal_matrix(N, seed=11)
    return c, to_np(c).astype(np.int32)


@pytest.mark.parametrize("expr", list(EXPRS))
def test_closure_is_lazy_and_structured(expr, causal):
    c, a = causal
    m = EXPRS[expr][0](c)
    assert m.is_lazy and m.dtype == "int32"
    expected_cls = pc.SymmetricMatrix if "+" in expr else pc.AntiSymmetricMatrix
    assert type(m) is expected_cls
    with eager():
        e = EXPRS[expr][0](c)
    assert not e.is_lazy and type(e) is expected_cls


@pytest.mark.parametrize("expr", list(EXPRS))
@pytest.mark.parametrize("consumer", list(CONSUMERS))
def test_closure_consumers_match_eager_and_numpy(expr, consumer, causal):
    c, a = causal
    pc_expr, np_expr = EXPRS[expr]
    got = CONSUMERS[consumer](pc_expr(c))
    with eager():
        ref = CONSUMERS[consumer](pc_expr(c))
    np.testing.assert_allclose(np.asarray(got, dtype=float), np.asarray(ref, dtype=float), rtol=1e-12, atol=1e-12)
    if consumer == "to_numpy":
        np.testing.assert_array_equal(got, np_expr(a))


def test_closure_matvec_uses_the_bits(causal):
    c, a = causal
    d = c - c.T
    native._debug_clear_kernel_trace()
    y = d @ pc.vector(np.ones(N))
    assert native._debug_last_kernel_trace() == "cpu.matvec.bit_closure"
    np.testing.assert_allclose(to_np(y), (a - a.T) @ np.ones(N))


def test_arnoldi_on_closure_matches_dense(causal):
    c, a = causal
    d = c - c.T
    w = np.sort_complex(to_np(pc.eigvals_arnoldi(d, 4, 16, 1e-10)))
    ref = np.linalg.eigvals((a - a.T).astype(float))
    ref = ref[np.argsort(-np.abs(ref), kind="stable")][:4]
    np.testing.assert_allclose(np.sort(np.abs(w)), np.sort(np.abs(ref)), rtol=1e-8)


def test_writes_materialize_in_place_and_never_leak(causal):
    c, a = causal
    d = c - c.T
    e = c - c.T
    d[0, 2] = 5
    assert d[0, 2] == 5 and d[2, 0] == -5 and not d.is_lazy
    np.testing.assert_array_equal(to_np(e), a - a.T)  # sibling closure unaffected
    np.testing.assert_array_equal(to_np(c).astype(np.int32), a)  # source unaffected


def test_source_mutation_does_not_leak_into_closure():
    c = pc.causal_matrix(8, seed=2)
    a = to_np(c).astype(np.int32)
    d = c - c.T
    c[0, 7] = not bool(c[0, 7])
    np.testing.assert_array_equal(to_np(d), a - a.T)


def test_closure_save_load(causal, tmp_path):
    c, a = causal
    d = c - c.T
    p = tmp_path / "d.pycauset"
    pc.save(d, p)
    loaded = pc.load(p)
    assert type(loaded) is pc.AntiSymmetricMatrix and loaded.dtype == "int32"
    np.testing.assert_array_equal(to_np(loaded), a - a.T)
