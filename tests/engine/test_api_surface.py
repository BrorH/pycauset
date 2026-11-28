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


@pytest.mark.parametrize("data", [[1.5, 2.0], [1 + 2j, 3], [True, False], [1, 2, 3]])
def test_vector_from_python_list(data):
    v = pc.vector(data)
    np.testing.assert_array_equal(to_np(v), np.array(data))
    if not all(type(x) is int for x in data):  # Python ints: int32 by design (F-103)
        assert np.asarray(v).dtype == np.array(data).dtype


@pytest.mark.parametrize("data", [[[1.0, 2], [3, 4]], [[1j, 0], [0, 1]], [[True, False], [False, True]]])
def test_matrix_from_nested_list(data):
    m = pc.matrix(data)
    np.testing.assert_array_equal(to_np(m), np.array(data))
    assert np.asarray(m).dtype == np.array(data).dtype


@pytest.mark.parametrize("fn,np_fn", [(pc.zeros, np.zeros), (pc.ones, np.ones)])
@pytest.mark.parametrize("dtype", ["float64", "int16", "complex_float32"])
def test_filled_vector_constructors(fn, np_fn, dtype):
    np_dtype = {"complex_float32": "complex64"}.get(dtype, dtype)
    v = fn(4, dtype=dtype)
    assert isinstance(v, pc.VectorBase) and v.dtype == dtype
    np.testing.assert_array_equal(to_np(v), np_fn(4, dtype=np_dtype))


def test_dot_vector_matrix_and_matrix_vector():
    v, m = np.array([1.0, 2.0]), np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(to_np(pc.dot(pc.vector(v), pc.matrix(m))), np.dot(v, m))
    np.testing.assert_array_equal(to_np(pc.dot(pc.matrix(m), pc.vector(v))), np.dot(m, v))


@pytest.mark.parametrize("fn,arg,match", [
    (pc.symmetric, np.ones((2, 3)), "square"),
    (pc.symmetric, np.array([[1.0, 2.0], [3.0, 4.0]]), "not symmetric"),
    (pc.antisymmetric, np.array([[1.0, 2.0], [-2.0, 0.0]]), "antisymmetric"),
    (pc.hermitian, np.array([[1, 2j], [2j, 1]]), "not Hermitian"),
    (pc.diagonal, np.ones((2, 3)), "1D vector or a 2D square"),
])
def test_structured_constructors_reject_wrong_input(fn, arg, match):
    with pytest.raises(ValueError, match=match):
        fn(arg)


def test_hermitian_of_real_input_is_complex():
    """Documented on the hermitian page: a real packed type is pycauset.symmetric."""
    h = pc.hermitian(np.eye(2))
    assert h.dtype == "complex_float64"
    np.testing.assert_array_equal(to_np(h), np.eye(2))


@pytest.mark.parametrize("fn", [pc.matrix, pc.vector])
@pytest.mark.parametrize("bad", [5, 2.0])
def test_constructors_take_data_not_sizes(fn, bad):
    with pytest.raises(TypeError, match="constructs from data"):
        fn(bad)


def test_vector_of_a_vector_is_itself():
    v = pc.vector([1.0, 2.0])
    assert pc.vector(v) is v
    with pytest.raises(TypeError):
        pc.vector(v, dtype="int32")


@pytest.mark.parametrize("fn,np_fn", [(pc.bitwise_and, np.bitwise_and), (pc.bitwise_or, np.bitwise_or),
                                      (pc.bitwise_xor, np.bitwise_xor)])
def test_bitwise_functions(fn, np_fn):
    x = np.array([[True, False], [True, True]])
    y = np.array([[True, True], [False, True]])
    np.testing.assert_array_equal(to_np(fn(pc.matrix(x), pc.matrix(y))), np_fn(x, y))


def test_matrix_power_negative():
    a = np.array([[2.0, 1.0], [0.0, 3.0]])
    np.testing.assert_allclose(to_np(pc.matrix_power(pc.matrix(a), -2)), np.linalg.matrix_power(a, -2))


def test_compute_k_is_c_times_inverse_of_a_i_plus_c():
    c = np.triu(np.ones((4, 4)), 1)
    k = pc.compute_k(pc.causal_matrix(c.astype(bool)), 2.0)
    np.testing.assert_allclose(to_np(k), c @ np.linalg.inv(2.0 * np.eye(4) + c), atol=1e-12)


def test_solve_triangular_requires_a_triangular_class():
    with pytest.raises(ValueError, match="TriangularMatrix"):
        pc.solve_triangular(pc.matrix(np.ones((2, 2))), pc.matrix(np.ones((2, 1))))
    t = pc.TriangularMatrix(2, dtype="float64", has_diagonal=True)
    t[0, 0], t[0, 1], t[1, 1] = 2.0, 1.0, 4.0
    np.testing.assert_allclose(to_np(pc.solve_triangular(t, pc.matrix(np.ones((2, 1))))),
                               np.linalg.solve([[2.0, 1.0], [0.0, 4.0]], np.ones((2, 1))))


def test_matrix_from_ndarray_with_dtype():
    m = pc.matrix(np.arange(4.0).reshape(2, 2), dtype="float32")
    assert m.dtype == "float32"
    np.testing.assert_array_equal(to_np(m), np.arange(4.0).reshape(2, 2))



def test_matrix_from_nested_list_with_dtype_and_bit_alias():
    assert pc.matrix([[1, 0], [0, 1]], dtype="float32").dtype == "float32"
    assert pc.matrix([[1, 0], [0, 1]], dtype="bit").dtype == "bool"


def test_matrix_of_existing_objects():
    m = pc.matrix(np.eye(2))
    assert pc.matrix(m) is m
    with pytest.raises(TypeError, match="already a matrix"):
        pc.matrix(m, dtype="int32")
    with pytest.raises(TypeError, match="1D or 2D"):
        pc.matrix(np.zeros((2, 2, 2)).tolist())
    assert isinstance(pc.matrix([1.0, 2.0]), pc.VectorBase)  # 1-D data: a vector, as np.array


def test_causal_matrix_of_a_size_without_population():
    c = pc.causal_matrix(5, populate=False)
    assert c.shape == (5, 5) and c.dtype == "bool"
    np.testing.assert_array_equal(to_np(c), np.zeros((5, 5), dtype=bool))



def test_str_shows_class_shape_and_elides_large_matrices():
    s = str(pc.matrix(np.arange(400).reshape(20, 20)))
    lines = s.splitlines()
    assert lines[0] == "DenseMatrix(shape=(20, 20))"
    rows = [ln for ln in lines if ln.startswith(" [")]
    assert len(rows) == 8 and " ..." in s  # 4 edge rows each side, 4 edge items per row
    assert rows[0] == " [0 1 2 3 ... 16 17 18 19]" and rows[-1].endswith("396 397 398 399]")
    small = str(pc.causal_matrix(np.triu(np.ones((3, 3), dtype=bool), 1))).splitlines()
    assert small == ["TriangularMatrix(shape=(3, 3))", "[", " [0 1 1]", " [0 0 1]", " [0 0 0]", "]"]
    assert repr(pc.matrix(np.eye(2))) == "<DenseMatrix shape=(2, 2) dtype=float64>"


@pytest.mark.parametrize("p", [0.5, 0.3])
def test_random_causal_triangle(p):
    """p = 0.5 fills whole words of random bits; other p draws each bit (Bernoulli)."""
    a = pc.TriangularMatrix.random(200, p=p, seed=5)
    x = np.asarray(a).astype(bool)
    assert np.array_equal(x, np.asarray(pc.TriangularMatrix.random(200, p=p, seed=5)))  # seeded
    assert not np.tril(x).any()  # strictly upper
    assert abs(x[np.triu_indices(200, 1)].mean() - p) < 0.02


def test_transpose_method_equals_T():
    a = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(to_np(pc.matrix(a).transpose()), a.T)
    np.testing.assert_array_equal(to_np(pc.matrix(a).T), a.T)


def test_unary_plus_and_set_identity():
    a = np.arange(4.0).reshape(2, 2)
    np.testing.assert_array_equal(to_np(+pc.matrix(a)), +a)
    m = pc.DenseMatrix(3, dtype="int16")
    m.set_identity()
    np.testing.assert_array_equal(to_np(m), np.eye(3, dtype=np.int16))


def test_dense_bool_random():
    a = pc.DenseMatrix.random(100, 80, p=0.2, seed=9)
    x = np.asarray(a).astype(bool)
    assert x.shape == (100, 80) and a.dtype == "bool"
    assert np.array_equal(x, np.asarray(pc.DenseMatrix.random(100, 80, p=0.2, seed=9)))
    assert abs(x.mean() - 0.2) < 0.03


def test_vector_conjugate_and_writes_into_views():
    v = pc.vector(np.array([1 + 2j, 3 - 1j]))
    np.testing.assert_array_equal(to_np(v.conjugate()), np.conj([1 + 2j, 3 - 1j]))
    w = pc.vector(np.array([1.0, 2.0, 3.0])) * 2.0  # a scaled view
    w[0] = 10.0  # the write folds the view into storage first; the other values are kept
    np.testing.assert_array_equal(to_np(w), [10.0, 4.0, 6.0])


def test_every_public_name_has_a_doc_page():
    """Every name in `pycauset.__all__` has a page `pycauset.<name>.md` (the submodules
    `spacetime`, `synthetic` and `vis` have their own pages). Internal native helpers start
    with an underscore so they stay out of `__all__`."""
    from pathlib import Path

    docs = Path(__file__).resolve().parents[2] / "documentation" / "docs"
    pages = {p.stem for p in docs.rglob("*.md")}
    missing = [n for n in sorted(pc.__all__) if f"pycauset.{n}" not in pages and n not in ("spacetime", "synthetic", "vis")]
    assert not missing, f"public names without a doc page: {missing}"


@pytest.mark.parametrize("dtype", ["float64", "int32", "complex_float64"])
def test_antisymmetric_storage_writes(dtype):
    """Writing below the diagonal stores the negated value above it; a non-zero diagonal
    raises, zero is accepted (moved from the retired tests/test_symmetric_matrix.cpp)."""
    a = pc.AntiSymmetricMatrix(4, dtype=dtype)
    a[2, 1] = 3
    assert a[1, 2] == -3 and a[2, 1] == 3
    with pytest.raises(ValueError, match="must be zero"):
        a[0, 0] = 1
    a[0, 0] = 0
    s = pc.SymmetricMatrix(4, dtype=dtype)
    s[3, 0] = 5
    assert s[0, 3] == 5


def test_stub_declares_every_public_name():
    """`__init__.pyi` overrides the module for type checkers: a public name it lacks is an
    error in IDEs on correct code (F-007)."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(pc.__file__).parent / "__init__.pyi").read_text())
    declared = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            declared.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared.add(node.target.id)
        elif isinstance(node, ast.Assign):
            declared.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.ImportFrom):
            declared.update(a.asname or a.name for a in node.names)
    missing = sorted(set(pc.__all__) - declared)
    assert not missing, f"public names missing from __init__.pyi: {missing}"


def test_stub_declares_only_what_exists():
    """The stub listed methods that were removed (MatrixBase.eig / matmul / determinant /
    multiply, the `properties` layer): a type checker accepted calls that raise."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(pc.__file__).parent / "__init__.pyi").read_text())
    stale = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and hasattr(pc, node.name):
            cls = getattr(pc, node.name)
            stale += [f"{node.name}.{f.name}" for f in node.body
                      if isinstance(f, ast.FunctionDef) and f.name != "__getattr__" and not hasattr(cls, f.name)]
    assert not stale, f"declared in __init__.pyi but missing: {stale}"


def test_type_stub_signatures_match_the_functions():
    """__init__.pyi's module-level functions name the same parameters as the real ones
    (13 were `(*args, **kwargs)`, F-007)."""
    import ast
    import inspect
    from pathlib import Path

    stub = ast.parse((Path(pc.__file__).parent / "__init__.pyi").read_text())
    checked = 0
    for node in stub.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        fn = getattr(pc, node.name, None)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue  # native builtins without a signature
        a = node.args
        stub_names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
        real = [q.name for q in sig.parameters.values() if q.kind not in (q.VAR_POSITIONAL, q.VAR_KEYWORD)]
        assert stub_names == real, f"{node.name}: stub {stub_names}, real {real}"
        checked += 1
    assert checked > 50
