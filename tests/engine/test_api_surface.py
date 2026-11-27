"""Thin public functions the conformance grid and atlas do not reach: each one is called
once and compared with NumPy (moved from the old suite while retiring it, TESTS.md §4)."""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np


@pytest.mark.parametrize("fn,np_fn", [(pc.zeros, np.zeros), (pc.ones, np.ones)])
@pytest.mark.parametrize("dtype", ["float64", "float32", "int16", "complex_float64", "bool"])
@pytest.mark.parametrize("shape", [(3, 3), (2, 5)])
def test_filled_constructors(fn, np_fn, dtype, shape):
    np_dtype = {"complex_float64": "complex128"}.get(dtype, dtype)
    m = fn(shape, dtype=dtype)
    assert m.shape == shape and m.dtype == dtype
    np.testing.assert_array_equal(to_np(m), np_fn(shape, dtype=np_dtype))


def test_empty_has_shape_and_dtype():
    m = pc.empty((4, 2), dtype="int32")
    assert m.shape == (4, 2) and m.dtype == "int32"


@pytest.mark.parametrize("arg,shape", [(3, (3, 3)), ([2, 4], (2, 4))])
def test_identity_from_size(arg, shape):
    np.testing.assert_array_equal(to_np(pc.identity(arg)), np.eye(*shape))


def test_identity_like_a_matrix_or_vector():
    assert pc.identity(pc.matrix(np.ones((2, 5)))).shape == (2, 5)
    assert pc.identity(pc.vector(np.ones(3))).shape == (3, 3)
    with pytest.raises(TypeError):
        pc.identity(True)


def test_matmul_function_equals_operator():
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(12.0).reshape(3, 4)
    np.testing.assert_allclose(to_np(pc.matmul(pc.matrix(a), pc.matrix(b))), a @ b)


@pytest.mark.parametrize("full", [True, False])
def test_svd_reconstructs(full):
    a = np.random.default_rng(0).standard_normal((5, 3))
    u, s, vt = pc.svd(pc.matrix(a), full_matrices=full)
    nu, ns, nvt = np.linalg.svd(a, full_matrices=full)
    assert u.shape == nu.shape and vt.shape == nvt.shape
    np.testing.assert_allclose(to_np(s), ns, atol=1e-12)
    uu, vv = to_np(u), to_np(vt)
    np.testing.assert_allclose(uu[:, :3] * to_np(s) @ vv[:3], a, atol=1e-12)


def test_bitwise_not_function():
    x = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(to_np(pc.bitwise_not(pc.matrix(x))), ~x)


@pytest.mark.parametrize("fn", [pc.eigvals_skew, pc.eig_skew])
def test_skew_eigen_argument_checks(fn):
    a = pc.matrix(np.array([[0.0, 1.0], [-1.0, 0.0]]))
    with pytest.raises(ValueError, match="positive"):
        fn(a, 0)
    with pytest.raises(ValueError, match="square"):
        fn(pc.matrix(np.ones((2, 3))), 1)


def test_setitem_accepts_numpy_scalars_and_slices():
    m = pc.matrix(np.zeros((3, 3)))
    m[0, 0] = np.float32(1.5)  # NumPy scalar: converted to a Python scalar
    m[1, :] = np.array([1.0, 2.0, 3.0])  # slice: the native __setitem__
    m[-1, -1] = 7.0  # negative index: the native __setitem__
    want = np.zeros((3, 3))
    want[0, 0], want[1, :], want[-1, -1] = 1.5, [1.0, 2.0, 3.0], 7.0
    np.testing.assert_array_equal(to_np(m), want)


def test_pc_bool_is_a_dtype_token():
    assert pc.bool == "bool"
    assert pc.matrix(np.zeros((2, 2)), dtype=pc.bool).dtype == "bool"
