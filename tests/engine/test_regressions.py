"""One regression test per finding in review/FINDINGS.md, named test_F<id>_<what>.

Open findings are marked xfail(strict=True): the fix makes the test pass, which makes
strict-xfail fail, which forces whoever fixed it to remove the marker and set the
finding's status.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

native = pc._native

open_bug = lambda fid: pytest.mark.xfail(strict=True, reason=f"{fid} open (review/FINDINGS.md)")  # noqa: E731


# ---------------------------------------------------------------- F-001
@pytest.mark.parametrize("scalar", [3, 3.0, 0, -1])
def test_F001_int32_scalar_multiply_is_applied(scalar):
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    r = pc.matrix(a) * scalar
    assert r[0, 1] == 2 * scalar
    np.testing.assert_array_equal(to_np(r), a * scalar)


# ---------------------------------------------------------------- F-002
def test_F002_unary_minus():
    a = np.array([[1.0, -2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(to_np(-pc.matrix(a)), -a)


@pytest.mark.parametrize("dtype", ["int32", "float64"])
def test_F002_int_scalar_on_the_left(dtype):
    a = np.array([[1, 2], [3, 4]], dtype=dtype)
    np.testing.assert_array_equal(to_np(3 * pc.matrix(a)), 3 * a)


def test_F002_int32_minus_int():
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    np.testing.assert_array_equal(to_np(pc.matrix(a) - 1), a - 1)


# ---------------------------------------------------------------- F-003
def test_F003_cond_default_matches_numpy():
    a = np.array([[4.0, 1.0], [2.0, 3.0]])
    assert pc.cond(pc.matrix(a)) == pytest.approx(np.linalg.cond(a))


@pytest.mark.parametrize("p", [None, 2, -2, 1, -1, np.inf, -np.inf, "fro"])
def test_F003_cond_all_norms_match_numpy(p, rng):
    a = rng.standard_normal((6, 6)) + 3 * np.eye(6)
    assert pc.cond(pc.matrix(a), p) == pytest.approx(np.linalg.cond(a, p), rel=1e-10)


def test_F003_cond_singular_is_inf():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert np.isinf(pc.cond(pc.matrix(a), 1)) and np.isinf(np.linalg.cond(a, 1))


# ---------------------------------------------------------------- F-024
@pytest.mark.parametrize("disk", [False, True])
def test_F024_eig_results_are_independent(disk, force_disk, rng):
    if not disk:
        pc.set_memory_threshold(8 * 1024**3)
    a, b = rng.standard_normal((6, 6)), rng.standard_normal((6, 6))
    w1, _ = pc.eig(pc.matrix(a))
    first = to_np(w1).copy()
    w2, _ = pc.eig(pc.matrix(b))
    np.testing.assert_array_equal(to_np(w1), first)
    assert w1.get_backing_file() != w2.get_backing_file() or w1.get_backing_file() == ":memory:"
    np.testing.assert_allclose(np.sort_complex(first), np.sort_complex(np.linalg.eigvals(a)), atol=1e-10)


def test_F024_eig_writes_nothing_into_cwd(_isolated_cwd, rng):
    pc.eig(pc.matrix(rng.standard_normal((4, 4))))
    assert not list(_isolated_cwd.iterdir())


# ---------------------------------------------------------------- F-028
# Policy: guesses may inform (warnings), never decide (results). The sampled int32
# matmul preflight is advisory; whether or not it fires, overflow detection is exact.
@pytest.mark.parametrize("big_at", [(0, 0), (57, 113)])  # sampled corner vs. unsampled spot
def test_F028_int32_matmul_overflow_detection_is_exact(big_at):
    a = np.zeros((200, 200), dtype=np.int32)
    a[big_at] = 50_000
    b = pc.matrix(np.full((200, 200), 50_000, dtype=np.int32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(OverflowError):
            pc.matrix(a) @ b


def test_F028_advisory_warning_never_changes_the_result():
    rng = np.random.default_rng(0)
    a = rng.integers(-3, 4, size=(200, 200)).astype(np.int32)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        got = to_np(pc.matrix(a) @ pc.matrix(a))
    np.testing.assert_array_equal(got, a.astype(np.int64) @ a.astype(np.int64))


# ---------------------------------------------------------------- F-006
# Integer elementwise + - * wrap (two's complement, like NumPy) without relying on
# signed-overflow UB. Values are checked here; UB-freedom is checked by the UBSan job.
@pytest.mark.parametrize(
    "dtype, a, b, op",
    [
        ("int32", 2**31 - 1, 1, "add"),
        ("int32", -(2**31), 1, "sub"),
        ("int32", 46341, 46341, "mul"),
        ("int64", 2**63 - 1, 1, "add"),
        ("int16", 32767, 1, "add"),
        ("int8", 127, 1, "add"),
        ("uint16", 65535, 65535, "mul"),   # promoted-to-int multiply: classic UB trap
        ("uint8", 255, 1, "add"),
    ],
)
def test_F006_integer_elementwise_wraps_like_numpy(dtype, a, b, op):
    x = np.full((3, 3), a, dtype=dtype)
    y = np.full((3, 3), b, dtype=dtype)
    with np.errstate(over="ignore"):
        expected = {"add": x + y, "sub": x - y, "mul": x * y}[op]
    px, py_ = pc.matrix(x), pc.matrix(y)
    got = {"add": lambda: px + py_, "sub": lambda: px - py_, "mul": lambda: px * py_}[op]()
    np.testing.assert_array_equal(to_np(got), expected)


# ---------------------------------------------------------------- F-025
def test_F025_save_and_load_empty_matrix(tmp_path):
    m = pc.zeros((0, 0), dtype="float64")
    path = tmp_path / "empty.pycauset"
    pc.save(m, path)
    loaded = pc.load(path)
    assert to_np(loaded).shape == (0, 0)


# ---------------------------------------------------------------- F-034
def test_F034_int_plus_float16_does_not_overflow_to_inf():
    a = pc.matrix(np.full((2, 2), 100_000, dtype=np.int32))
    b = pc.matrix(np.zeros((2, 2), dtype=np.float16))
    out = to_np(a + b)
    assert np.isfinite(out).all() and out[0, 0] == 100_000


def test_F034_int32_division_keeps_precision():
    a = np.full((2, 2), 16_777_217, dtype=np.int32)  # 2**24 + 1: not representable in float32
    out = to_np(pc.matrix(a) / pc.matrix(np.ones((2, 2), dtype=np.int32)))
    assert out[0, 0] == 16_777_217


# ---------------------------------------------------------------- F-037
@pytest.mark.parametrize("dtype", ["float64", "int32", "bool", "complex128"])
def test_F037_save_load_keeps_transposed_views(dtype, tmp_path):
    a = np.arange(12).reshape(3, 4).astype(dtype)
    m = pc.matrix(a).T  # 4x3 view of 3x4 storage
    pc.save(m, tmp_path / "t.pycauset")
    loaded = pc.load(tmp_path / "t.pycauset")
    np.testing.assert_array_equal(to_np(loaded), a.T)


# ---------------------------------------------------------------- F-038 / F-039
def test_F038_float_arithmetic_returns_real_matrices():
    a = pc.matrix(np.ones((3, 3)))
    for r in (a + a, a - a, a / 2.0, a * 2.5, 2.0 * a, (a + a) * 2.0 - a):
        assert type(r).__name__ != "LazyMatrix"
        pc.sum(r), r.T  # full matrix API available


@pytest.mark.parametrize("expr", ["C+C.T", "C-C.T", "C*C.T", "C+C", "C*C", "C.T+C.T"])
def test_F039_triangular_orientation_is_respected(expr):
    c = pc.causal_matrix(12, populate=True, seed=5)
    cn = to_np(c).astype(np.int64)
    env_pc, env_np = {"C": c}, {"C": cn}
    got = to_np(eval(expr, {}, env_pc)).astype(np.int64)
    np.testing.assert_array_equal(got, eval(expr, {}, env_np))


@pytest.mark.parametrize("op", ["+", "-", "*"])
def test_F039_float_upper_with_lower_triangle(op, rng):
    u = np.triu(rng.standard_normal((6, 6)), 1)
    tu = pc.matrix(u)
    got = to_np(eval(f"tu {op} tu.T", {}, {"tu": tu}))
    np.testing.assert_allclose(got, eval(f"u {op} u.T", {}, {"u": u}))


# ---------------------------------------------------------------- F-041 (TriangularMatrix)
def test_triangular_matrix_is_the_native_structure_class():
    # F-041a, superseded by the structure classes (RESTRUCTURE 3.1 item 4): pc.TriangularMatrix
    # is the real native class for every dtype; a causal matrix is one of dtype "bool".
    c = pc.causal_matrix(5, populate=True)
    assert type(c) is pc.TriangularMatrix and c.dtype == "bool"
    t = pc.TriangularMatrix(4, dtype="int16")
    t[0, 3] = -7
    assert type(t) is pc.TriangularMatrix and t.dtype == "int16" and t[0, 3] == -7 and t[3, 0] == 0


# Structured results for every dtype the structure supports (F-032 follow-up, PLAN 4.1d).
@pytest.mark.parametrize("kind,dtype", [
    ("TriangularMatrix", "int16"), ("TriangularMatrix", "float32"), ("TriangularMatrix", "uint8"),
    ("SymmetricMatrix", "int8"), ("SymmetricMatrix", "float32"), ("AntiSymmetricMatrix", "int64"),
    ("DiagonalMatrix", "uint8"), ("DiagonalMatrix", "int64"),
])
def test_structured_results_keep_structure_for_every_dtype(kind, dtype):
    cls = getattr(pc, kind)
    a, b = cls(5, dtype=dtype), cls(5, dtype=dtype)
    a[0, 0 if kind == "DiagonalMatrix" else 3] = 2
    b[1, 1 if kind == "DiagonalMatrix" else 4] = 3
    for res, ref in [(a + b, to_np(a) + to_np(b)), (a - b, to_np(a) - to_np(b))]:
        assert type(res) is cls and res.dtype == dtype
        np.testing.assert_array_equal(to_np(res), ref)
    if kind == "TriangularMatrix":
        assert type(a * 3) is cls  # scaling keeps the structure
        if not dtype.startswith("float"):
            assert type(a + 0) is cls  # exact integer kernel: zero-preserving ops keep it too


# L2a: A ± A^T is stored packed (exactly symmetric / antisymmetric), values unchanged.
@pytest.mark.parametrize("src", ["causal", "float", "int16"])
def test_transpose_pair_sums_are_packed(src, rng):
    if src == "causal":
        m = pc.causal_matrix(12, seed=4)
        a = to_np(m).astype(np.int32)
    elif src == "float":
        a = rng.standard_normal((12, 12))
        m = pc.matrix(a)
    else:
        a = rng.integers(-50, 50, (12, 12)).astype(np.int16)
        m = pc.matrix(a)
    s, d, e = m + m.T, m - m.T, m.T - m
    assert type(s) is pc.SymmetricMatrix and type(d) is pc.AntiSymmetricMatrix and type(e) is pc.AntiSymmetricMatrix
    np.testing.assert_array_equal(to_np(s), a + a.T)
    np.testing.assert_array_equal(to_np(d), a - a.T)
    np.testing.assert_array_equal(to_np(e), a.T - a)


def test_transpose_pair_with_nonfinite_diagonal_stays_dense():
    a = np.eye(4)
    a[2, 2] = np.inf
    m = pc.matrix(a)
    d = m - m.T
    assert type(d) is pc.DenseMatrix  # inf - inf = nan on the diagonal
    np.testing.assert_array_equal(to_np(d), a - a.T)


def test_pauli_jordan_like_spectrum_from_packed_result():
    c = pc.causal_matrix(60, seed=7)
    a = to_np(c).astype(np.float64)
    d = c - c.T
    w = np.sort(np.abs(to_np(pc.eigvals_skew(d, 6))))
    ref = np.sort(np.abs(np.linalg.eigvals(a - a.T)))[-6:]
    np.testing.assert_allclose(np.sort(w), np.sort(ref), rtol=1e-8)


# F-053: links() used a NumPy uint8 product; path counts of 256 wrapped to 0 -> false links.
def test_F053_links_of_a_long_chain():
    c = pc.synthetic.chain(300)
    links = c.links()
    assert int(links.sum()) == 299 and not links[0, 257]


@pytest.mark.parametrize("n,seed", [(150, 1), (400, 7)])
def test_links_match_the_definition(n, seed):
    cs = pc.causet(n=n, seed=seed)
    c = np.asarray(cs.C, dtype=bool)
    ci = c.astype(np.int64)
    np.testing.assert_array_equal(cs.links(), c & ~((ci @ ci) > 0))


# F-055: integer matmul runs exactly through float64 BLAS when the measured bound
# k * max|A| * max|B| < 2**53 holds; otherwise the exact integer kernels.
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint16", "uint64"])
def test_F055_int_matmul_via_blas_is_exact(dtype, rng):
    a = rng.integers(0, 3, (37, 23)).astype(dtype)
    b = rng.integers(0, 3, (23, 19)).astype(dtype)
    native._debug_clear_kernel_trace()
    c = pc.matrix(a) @ pc.matrix(b)
    assert native._debug_last_kernel_trace() == "cpu.matmul.int_via_dgemm"
    np.testing.assert_array_equal(to_np(c), (a.astype(np.int64) @ b.astype(np.int64)).astype(dtype))


def test_F055_beyond_the_exact_bound_uses_integer_kernel():
    a = np.full((3, 3), 2**40, dtype=np.int64)
    b = np.full((3, 3), 2**20, dtype=np.int64)
    native._debug_clear_kernel_trace()
    c = pc.matrix(a) @ pc.matrix(b)
    assert native._debug_last_kernel_trace() != "cpu.matmul.int_via_dgemm"
    np.testing.assert_array_equal(to_np(c), a @ b)


def test_F055_overflow_still_raises():
    x = pc.matrix(np.full((2, 2), 200, np.int16))
    with pytest.raises(OverflowError):
        x @ x


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint64", "bool"])
def test_F051_integer_dot_is_an_exact_int(dtype, rng):
    a = rng.integers(0, 2 if dtype == "bool" else 100, 50).astype(dtype)
    b = rng.integers(0, 2 if dtype == "bool" else 100, 50).astype(dtype)
    exact = int(np.dot(a.astype(object), b.astype(object)))
    for got in (pc.dot(pc.vector(a), pc.vector(b)), pc.vector(a) @ pc.vector(b)):
        assert type(got) is int and got == exact  # reductions never wrap (DType System 5.1)


def test_F051_integer_dot_applies_the_view_factor_and_is_exact_beyond_2_53():
    v = pc.vector(np.arange(1, 6, dtype=np.int32))
    w = pc.vector(np.arange(1, 6, dtype=np.int32))
    w.scalar = 3  # an integer view: logical elements 3, 6, ...
    assert pc.dot(w, v) == v @ w == 165
    big = pc.vector(np.array([2**62, 3], dtype=np.int64))
    assert pc.dot(big, pc.vector(np.array([3, -1], dtype=np.int64))) == 3 * 2**62 - 3
    with pytest.raises(OverflowError):
        huge = pc.vector(np.full(8, 2**64 - 1, dtype=np.uint64))
        pc.dot(huge, huge)


@pytest.mark.parametrize("dtype", ["int8", "int16", "int64", "uint64", "bool", "float32", "float64", "complex128"])
def test_F051_sum_has_numpy_result_kind_and_is_exact(dtype, rng):
    a = rng.integers(0, 2 if dtype == "bool" else 90, (7, 5)).astype(dtype)
    expected_type = int if dtype.startswith(("int", "uint", "bool")) else complex if "complex" in dtype else float
    for obj, ref in ((pc.matrix(a), a), (pc.vector(a.ravel()), a.ravel())):
        got = pc.sum(obj)
        assert type(got) is expected_type
        assert got == (int(ref.astype(object).sum()) if expected_type is int else ref.sum())


def test_F051_integer_sums_beyond_2_53_and_of_views():
    big = pc.matrix(np.full((3, 3), 2**62 + 1, dtype=np.int64))
    assert pc.sum(big) == 9 * (2**62 + 1)
    v = pc.matrix(np.arange(4, dtype=np.int8).reshape(2, 2)) * 100  # elements wrap in int8
    assert pc.sum(v) == int((np.arange(4, dtype=np.int8) * np.int8(100)).astype(np.int64).sum())


@pytest.mark.parametrize("n", [1, 64, 65, 300])
def test_F051_causal_sums_count_relations(n):
    c = pc.causal_matrix(n, seed=3)
    rel = int(to_np(c).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert pc.sum(c) == pc.sum(c.T) == rel
        assert pc.sum(c + c.T) == 2 * rel and pc.sum(c - c.T) == 0
        assert pc.sum(3 * (c + c.T)) == 6 * rel


# ---------------------------------------------------------------- F-081
_F081_OPS = {
    "sum": lambda m: pc.sum(m),
    "trace": lambda m: m.trace(),
    "det": lambda m: pc.determinant(m),
    "norm": lambda m: pc.norm(m),
    "inv": lambda m: to_np(pc.invert(m)),
    "eigvalsh": lambda m: np.sort(to_np(pc.eigvalsh(m))),
}


@pytest.mark.parametrize("op", sorted(_F081_OPS))
@pytest.mark.parametrize("write", ["setitem", "set"])
def test_F081_results_follow_mutation(op, write):
    a = np.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]])
    m = pc.matrix(a)
    f = _F081_OPS[op]
    f(m)  # a first call must not pin the answer
    if write == "setitem":
        m[1, 1] = 9.0
    else:
        m.set(1, 1, 9.0)
    a[1, 1] = 9.0
    ref = {"sum": a.sum(), "trace": np.trace(a), "det": np.linalg.det(a), "norm": np.linalg.norm(a),
           "inv": np.linalg.inv(a), "eigvalsh": np.linalg.eigvalsh(a)}[op]
    np.testing.assert_allclose(f(m), ref, rtol=1e-12)


@pytest.mark.parametrize("flags", [{"is_zero": True}, {"is_identity": True}, {"diagonal_value": 7.0, "is_diagonal": True}])
def test_F081_properties_never_change_results(flags):
    a = np.array([[2.0, 0.0], [0.0, 3.0]])
    m = pc.matrix(a)
    try:
        m.properties.update(flags)
    except ValueError:
        pytest.skip("flags rejected as incompatible with the values")
    assert m.trace() == pytest.approx(5.0)
    assert pc.determinant(m) == pytest.approx(6.0)
    assert pc.sum(m) == pytest.approx(5.0)
