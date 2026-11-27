"""NumPy in, NumPy out, and the binding-level behaviours around them (TESTS.md layout:
test_interop.py). Moved from the old suite while retiring it, by C++ coverage: every case
here ran code that only an old test file reached."""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

A = np.arange(9.0).reshape(3, 3)


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
def test_matrix_op_numpy_2d(op):
    m, b = pc.matrix(A), A + 1.0
    np.testing.assert_allclose(to_np(eval(f"m {op} b")), eval(f"A {op} b"))


@pytest.mark.parametrize("rhs", [A + 1.0, np.arange(3.0)])
def test_matmul_with_numpy_operand(rhs):
    np.testing.assert_allclose(to_np(pc.matrix(A) @ rhs), A @ rhs)


@pytest.mark.parametrize("other", [np.arange(3.0)[None, :], np.arange(3.0)[:, None], np.arange(3.0)])
def test_broadcasting(other):
    got = pc.matrix(A) + (pc.matrix(other) if other.ndim == 2 else other)
    np.testing.assert_allclose(to_np(got), A + other)


def test_integer_array_indexing_get_and_set():
    rows, cols = np.array([0, 2]), np.array([1, 1])
    m = pc.matrix(A.copy())
    np.testing.assert_array_equal(to_np(m[rows, cols]).ravel(), A[rows, cols])
    n = A.copy()
    m[rows, cols] = 7.0
    n[rows, cols] = 7.0
    np.testing.assert_array_equal(to_np(m), n)


def test_boolean_mask_assignment():
    m, n = pc.matrix(A.copy()), A.copy()
    mask = np.array([True, False, True])
    m[mask, 0] = 4.0
    n[mask, 0] = 4.0
    np.testing.assert_array_equal(to_np(m), n)


def test_symmetric_assignment_mirrors():
    s = pc.symmetric(A + A.T)
    s[0, 1] = 9.0
    assert to_np(s)[1, 0] == 9.0


def test_element_conversion_errors_match_numpy():
    b = pc.matrix(np.zeros((2, 2), dtype=bool))
    with pytest.raises(TypeError):
        b[0, 0] = 2
    i8 = pc.matrix(np.zeros((2, 2), dtype=np.int8))
    with pytest.raises(OverflowError):
        i8[0, 0] = 1000
    with pytest.raises(OverflowError):
        np.zeros((2, 2), dtype=np.int8)[0, 0] = 1000  # NumPy raises too


def test_integer_scaled_view_reads_wrapped_value():
    m = pc.matrix(np.arange(4, dtype=np.int32).reshape(2, 2)) * 3
    assert float(m[1, 1]) == 9.0


def test_file_backed_export_needs_allow_huge(tmp_path):
    previous, old_dir = pc.get_memory_threshold(), pc._storage_root()
    pc.set_backing_dir(tmp_path)
    pc.set_memory_threshold(1)  # every new matrix is file-backed (a .tmp file)
    try:
        m = pc.matrix(np.ones((8, 8)))
        with pytest.raises(RuntimeError, match="file-backed"):
            np.asarray(m)
        np.testing.assert_array_equal(pc.to_numpy(m, allow_huge=True), np.ones((8, 8)))
    finally:
        pc.set_memory_threshold(previous)
        pc.set_backing_dir(old_dir)


def test_vector_indexing_and_repr():
    v = pc.vector(np.arange(4.0))
    assert v[-1] == 3.0 and v[0] == 0.0
    assert repr(v) == "<DenseVector shape=(4,) dtype=float64 transposed=False>"


def test_column_times_row_vector_is_outer():
    v, w = pc.vector(np.arange(4.0)), pc.vector(np.ones(3))
    np.testing.assert_array_equal(to_np(v @ w.T), np.outer(np.arange(4.0), np.ones(3)))


def test_solve_with_vector_rhs():
    a = A + 10 * np.eye(3)
    np.testing.assert_allclose(to_np(pc.solve(pc.matrix(a), pc.vector(np.ones(3)))), np.linalg.solve(a, np.ones(3)))


def test_eigen_methods_on_matrices():
    w, _ = pc.matrix(A).eig()
    np.testing.assert_allclose(np.sort_complex(to_np(w)), np.sort_complex(np.linalg.eigvals(A)), atol=1e-10)


def test_uint16_vector_import():
    v = pc.vector(np.arange(3, dtype=np.uint16))
    assert v.dtype == "uint16"
    np.testing.assert_array_equal(to_np(v), np.arange(3))


def test_set_num_threads_round_trip():
    previous = pc.get_num_threads()
    try:
        pc.set_num_threads(2)
        assert pc.get_num_threads() == 2
    finally:
        pc.set_num_threads(previous)


def test_matvec_promotion_is_reported():
    info = pc._debug_resolve_promotion("matvec", "float32", "float64")
    assert info["result_dtype"] == "float32" and info["float_underpromotion"]


def test_ellipsis_index():
    np.testing.assert_array_equal(to_np(pc.matrix(A)[...]), A[...])


def test_list_row_index():
    np.testing.assert_array_equal(to_np(pc.matrix(A)[[0, 2], :]), A[[0, 2], :])


@pytest.mark.parametrize("left", [np.arange(3.0), np.arange(6.0).reshape(2, 3)])
def test_ndarray_matmul_matrix(left):
    np.testing.assert_allclose(to_np(left @ pc.matrix(A)), left @ A)


def test_numpy_integer_scalars_index():
    m = pc.matrix(A)
    assert m[np.int64(1), np.int32(2)] == A[1, 2]


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
@pytest.mark.parametrize("reflected", [False, True])
def test_matrix_op_pycauset_vector_broadcasts(op, reflected):
    """F-102: a 1-D operand broadcasts as a row, as in NumPy."""
    m, v = pc.matrix(A + 1.0), pc.vector(np.arange(1.0, 4.0))
    x, y = (v, m) if reflected else (m, v)
    nx, ny = (np.arange(1.0, 4.0), A + 1.0) if reflected else (A + 1.0, np.arange(1.0, 4.0))
    np.testing.assert_allclose(to_np(eval(f"x {op} y")), eval(f"nx {op} ny"))


def test_assignment_into_structured_matrices_by_slice():
    t = pc.TriangularMatrix(3, dtype="float64", has_diagonal=True)
    t[0, :] = np.array([1.0, 2.0, 3.0])
    t[1, 1:] = 5.0
    np.testing.assert_array_equal(to_np(t), [[1, 2, 3], [0, 5, 5], [0, 0, 0]])
    s = pc.symmetric(np.zeros((3, 3)))
    s[0, 1:] = np.array([7.0, 8.0])
    np.testing.assert_array_equal(to_np(s)[1:, 0], [7.0, 8.0])  # mirrored
    with pytest.raises(ValueError, match="broadcast"):
        t[0, :] = np.array([1.0, 2.0])


@pytest.mark.parametrize("data", [np.array([1 + 2j, 3j]), np.array([3, -4], dtype=np.int16), np.array([True, False])])
def test_vector_element_types(data):
    v = pc.vector(data)
    assert [v[i] for i in range(len(data))] == data.tolist()
    assert type(v[0]) is type(data.tolist()[0])


def test_transposed_vector_repr_and_str():
    v = pc.vector(np.arange(3.0)).T
    assert repr(v) == str(v) == "<DenseVector shape=(1, 3) dtype=float64 transposed=True>"


def test_matvec_promotion_with_an_integer_side():
    assert pc._debug_resolve_promotion("matvec", "int32", "float32")["result_dtype"] == "float64"
    assert pc._debug_resolve_promotion("matvec", "int8", "float32")["result_dtype"] == "float32"


def test_large_import_is_exact():
    """Arrays of 1 MB and more take the parallel copy (numpy_import.hpp), strided or not."""
    x = np.random.default_rng(0).standard_normal((600, 600))
    np.testing.assert_array_equal(to_np(pc.matrix(x)), x)
    np.testing.assert_array_equal(to_np(pc.matrix(x[::2, ::-1])), x[::2, ::-1])


def test_incompatible_shapes_raise_like_numpy():
    with pytest.raises(ValueError, match="could not be broadcast"):
        pc.matrix(np.ones((2, 3))) + pc.matrix(np.ones((3, 2)))
    with pytest.raises(ValueError):
        np.ones((2, 3)) + np.ones((3, 2))


def test_solve_with_matrix_rhs():
    a, b = A + 10 * np.eye(3), np.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(to_np(pc.solve(pc.matrix(a), pc.matrix(b))), np.linalg.solve(a, b))


def test_eig_skew_odd_size_has_a_zero_eigenvalue():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 5))
    s = x - x.T
    vals, vecs = pc.eig_skew(pc.matrix(s), 5)
    w = to_np(vals)
    assert np.min(np.abs(w)) < 1e-10
    v = to_np(vecs)
    np.testing.assert_allclose(s @ v, v * w, atol=1e-9)


def test_paired_index_arrays_on_a_structured_matrix():
    s = pc.symmetric(np.zeros((3, 3)))
    s[np.array([0, 1]), np.array([2, 2])] = np.array([4.0, 5.0])
    np.testing.assert_array_equal(to_np(s)[2, :2], [4.0, 5.0])
    with pytest.raises(ValueError, match="same length"):
        s[np.array([0, 1]), np.array([2])] = 1.0


def test_eigh_method():
    h = A + A.T
    w, _ = pc.matrix(h).eigh()
    np.testing.assert_allclose(to_np(w), np.linalg.eigvalsh(h), atol=1e-10)


@pytest.mark.parametrize("mdt,vdt", [("int32", "float32"), ("float32", "int8"), ("int16", "int16")])
def test_matvec_dtype_matches_numpy(mdt, vdt):
    m = np.arange(9).reshape(3, 3).astype(mdt)
    v = np.arange(3).astype(vdt)
    got = pc.matrix(m) @ pc.vector(v)
    want = m @ v
    np.testing.assert_allclose(to_np(got), want)
    # F-034 (within-float under-promotion) is PyCauset's documented rule; others are NumPy's.
    assert np.asarray(got).dtype == want.dtype or ("float32" in (mdt, vdt) and np.asarray(got).dtype == np.float32)


def test_vecmat_promotion_is_reported():
    assert pc._debug_resolve_promotion("vecmat", "int32", "float32")["result_dtype"] == "float64"
    assert pc._debug_resolve_promotion("vecmat", "float32", "float32")["result_dtype"] == "float32"


def test_typed_element_accessors_honour_the_view():
    m = pc.matrix(np.array([[1, 2], [3, 100]], dtype=np.int8)) * 3  # integer factor wraps like NumPy
    assert m.get_element_as_double(1, 1) == float(np.int8(100) * np.int8(3))
    v = pc.vector(np.array([1 + 1j, 2j]))
    assert v.get_element_as_complex(1) == 2j
    assert v.conj().get_element_as_complex(0) == 1 - 1j


@pytest.mark.parametrize("op", ["matvec", "vecmat"])
def test_complex_matvec_promotion(op):
    # complex64 with a float32 side stays complex64; an int32 side needs complex128 (F-034).
    assert pc._debug_resolve_promotion(op, "complex_float32", "float32")["result_dtype"] == "complex_float32"
    assert pc._debug_resolve_promotion(op, "complex_float32", "int32")["result_dtype"] == "complex_float64"


def test_complex_matvec_matches_numpy():
    m = (np.arange(9).reshape(3, 3) + 1j).astype(np.complex64)
    v = np.arange(3, dtype=np.int32)
    np.testing.assert_allclose(to_np(pc.matrix(m) @ pc.vector(v)), m @ v, rtol=1e-6)
