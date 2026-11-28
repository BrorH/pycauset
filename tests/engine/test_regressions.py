"""One regression test per finding in review/FINDINGS.md, named test_F<id>_<what>.

Open findings are marked xfail(strict=True): the fix makes the test pass, which makes
strict-xfail fail, which forces whoever fixed it to remove the marker and set the
finding's status.
"""
from __future__ import annotations

import operator

import warnings

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

_BIN = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "@": operator.matmul}

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


# F-053: links used a NumPy uint8 product; path counts of 256 wrapped to 0 -> false links.
# The link matrix is now an engine bit kernel (math/CausalMatrix.cpp).
def _closure(c: np.ndarray) -> np.ndarray:
    """Transitive closure of a strictly upper boolean matrix (NumPy reference)."""
    r = c.copy()
    while True:
        nxt = r | ((r.astype(np.int64) @ r.astype(np.int64)) > 0)
        if np.array_equal(nxt, r):
            return r
        r = nxt


def test_F053_links_of_a_long_chain():
    chain = np.triu(np.ones((300, 300), dtype=bool), 1)
    links = np.asarray(native._transitive_reduction(pc.causal_matrix(chain))).astype(bool)
    assert int(links.sum()) == 299 and not links[0, 257]


@pytest.mark.parametrize("n,seed", [(150, 1), (400, 7)])
def test_links_match_the_definition(n, seed):
    rng = np.random.default_rng(seed)
    c = _closure(np.triu(rng.random((n, n)) < 3.0 / n, 1))
    ci = c.astype(np.int64)
    got = np.asarray(native._transitive_reduction(pc.causal_matrix(c))).astype(bool)
    np.testing.assert_array_equal(got, c & ~((ci @ ci) > 0))


def test_transitive_reduction_rejects_views():
    c = pc.causal_matrix(np.triu(np.ones((4, 4), dtype=bool), 1))
    with pytest.raises(TypeError):
        native._transitive_reduction(c.T)


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


def test_F093_loaded_file_is_a_snapshot(tmp_path):
    """Writing to a loaded matrix must not change the saved file (copy-on-write mapping)."""
    import subprocess
    import sys

    p = tmp_path / "snap.pycauset"
    pc.save(pc.matrix(np.arange(9.0).reshape(3, 3)), p)
    w = pc.load(p)
    w[0, 0] = 99.0
    assert w[0, 0] == 99.0  # the object sees its write
    assert pc.to_numpy(pc.load(p), allow_huge=True)[0, 0] == 0.0  # the file does not
    import os
    from pathlib import Path

    code = f"import pycauset as pc; print(pc.to_numpy(pc.load({str(p)!r}), allow_huge=True)[0, 0])"
    env = dict(os.environ, PYTHONPATH=str(Path(pc.__file__).resolve().parents[1]))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "0.0"  # nor from another process
    pc.save(w, p)  # saving over the file it was loaded from (atomic replace)
    assert pc.to_numpy(pc.load(p), allow_huge=True)[0, 0] == 99.0
    assert w[1, 1] == 4.0  # the live object still reads its mapping


@pytest.mark.parametrize("structure", ["dense", "triangular", "symmetric", "diagonal", "identity"])
def test_det_structure_rules_match_numpy(structure):
    rng = np.random.default_rng(3)
    a = rng.standard_normal((5, 5)) + 3 * np.eye(5)
    if structure == "triangular":
        a, m = np.triu(a), None
        m = pc.TriangularMatrix(5, dtype="float64", has_diagonal=True)
        pc._native._debug_fill(m, a)
    elif structure == "symmetric":
        a = a + a.T
        m = pc.symmetric(a)
    elif structure == "diagonal":
        a = np.diag(np.diag(a))
        m = pc.diagonal(np.diag(a))
    elif structure == "identity":
        a, m = np.eye(5), pc.IdentityMatrix(5)
    else:
        m = pc.matrix(a)
    for view, ref in ((m, a), (m.T * 2.5, a.T * 2.5)):
        assert pc.determinant(view) == pytest.approx(np.linalg.det(ref), rel=1e-10)
        s, ld = pc.slogdet(view)
        rs, rld = np.linalg.slogdet(ref)
        assert s == pytest.approx(rs) and ld == pytest.approx(rld, rel=1e-12)


def test_det_single_precision_like_numpy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert pc.determinant(pc.matrix(a)) == pytest.approx(float(np.linalg.det(a)), rel=1e-6)


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.complex128, bool])
def test_F097_views_of_non_square_and_windowed_matrices(dtype):
    """Scaled views of transposed non-square matrices and transposed windows kept the wrong
    shape / offsets (hand-built view copies, F-097)."""
    a = (np.arange(1, 37).reshape(6, 6) % 5).astype(dtype)
    m = pc.matrix(a)
    np.testing.assert_array_equal(to_np(m[1:4, 2:6].T), a[1:4, 2:6].T)
    if dtype is not bool:
        b = a[:2, :3]
        r = pc.matrix(b.copy())
        np.testing.assert_array_equal(to_np(r.T * 2), b.T * 2)
        np.testing.assert_array_equal(to_np(m[1:4, 2:6].T * 3), a[1:4, 2:6].T * 3)
    v = pc.vector(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(to_np(pc.IdentityMatrix(3) @ v), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("dtype", ["int16", "float32", "complex128", "bool", "uint8"])
def test_F101_matrix_power_zero_keeps_dtype(dtype):
    """a ** 0 is the identity in a's dtype, as NumPy's matrix_power (it was float64)."""
    x = np.eye(3, dtype=dtype)
    got = pc.matrix_power(pc.matrix(x), 0)
    assert np.asarray(got).dtype == np.linalg.matrix_power(x, 0).dtype
    np.testing.assert_array_equal(np.asarray(got), np.eye(3))


_ID_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint32", "float16", "float32", "float64",
              "complex64", "complex128"]


@pytest.mark.parametrize("dtype", _ID_DTYPES)
@pytest.mark.parametrize("op", ["+", "-", "*"])
def test_F099_identity_pair_every_dtype(dtype, op):
    """identity (+ - *) identity is a scaled identity for every dtype (it was float64 and int32
    only; the others went element by element)."""
    pc_dtype = {"complex64": "complex_float32", "complex128": "complex_float64"}.get(dtype, dtype)
    eye = pc.IdentityMatrix(4, dtype=pc_dtype)
    a, b = eye, eye
    na = nb = np.eye(4, dtype=dtype)
    if dtype.startswith(("float", "complex")):  # integer scaling writes out a dense matrix
        a, b = eye * 3, eye * 2
        na, nb = na * 3, nb * 2
    got = _BIN[op](a, b)
    want = _BIN[op](na, nb)
    assert type(got).__name__ == "IdentityMatrix"
    np.testing.assert_array_equal(to_np(got), want)


def test_F099_identity_pair_honours_conjugation():
    eye = pc.IdentityMatrix(3, dtype="complex_float64")
    a = (eye * (1 + 2j)).conj()
    b = eye * (3 - 1j)
    na, nb = np.conj(np.eye(3) * (1 + 2j)), np.eye(3) * (3 - 1j)
    for op in ("+", "-", "*"):
        np.testing.assert_array_equal(to_np(_BIN[op](a, b)), _BIN[op](na, nb))


def test_F099_identity_pair_integer_wraps_like_numpy():
    eye = pc.IdentityMatrix(2, dtype="uint8")
    x = eye
    nx = np.eye(2, dtype=np.uint8)
    for _ in range(9):  # 2^9 = 512 wraps to 0 in uint8
        x, nx = x + x, nx + nx
    np.testing.assert_array_equal(to_np(x), nx)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int16", "int64", "complex64"])
@pytest.mark.parametrize("step", [1, -1, 2, -2])
def test_vector_arith_with_numpy_operand(dtype, step):
    """`v + array` imports the array through the one shared importer (numpy_import.hpp,
    review 4.2: bind_vector had a diverging copy with an unsigned stride and a float32 ->
    float64 branch)."""
    base = np.arange(8).astype(dtype)
    x = base[::step]
    v = pc.vector(np.ones(len(x), dtype=dtype))
    got = v + x
    want = np.ones(len(x), dtype=dtype) + x
    assert np.asarray(got).dtype == want.dtype
    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("make", [lambda: pc.vector([1, 2]), lambda: pc.matrix([[1, 2], [3, 4]]),
                                  lambda: pc.vector(np.array([1.0, 2.0]), dtype=int),
                                  lambda: pc.zeros((2, 2), dtype="int")])
def test_F103_python_integers_default_to_int32(make):
    """Deliberate difference from NumPy (DType System §3.2)."""
    assert make().dtype == "int32"


def test_F103_typed_input_keeps_its_dtype():
    assert pc.vector(np.array([1, 2], dtype=np.int64)).dtype == "int64"
    assert pc.matrix(np.array([[1]], dtype=np.int8)).dtype == "int8"


def _tri(n, dtype, diag, seed):
    rng = np.random.default_rng(seed)
    ref = np.triu(rng.integers(1, 5, (n, n)), 0 if diag else 1).astype(dtype)
    m = pc.TriangularMatrix(n, dtype=dtype, has_diagonal=diag)
    for i in range(n):
        for j in range(i if diag else i + 1, n):
            m[i, j] = ref[i, j].item()
    return m, ref


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int16"])
@pytest.mark.parametrize("da,db,op,want_diag", [
    (True, True, "+", True), (False, True, "+", True), (False, False, "+", False),
    (True, True, "-", True), (True, True, "*", True), (False, True, "*", False),
])
def test_F099_triangular_results_keep_or_drop_the_diagonal(dtype, da, db, op, want_diag):
    """tri (+ - *) tri is a TriangularMatrix; the diagonal is kept exactly when the result
    can be non-zero there (it was a dense result whenever a diagonal was involved)."""
    a, ra = _tri(6, dtype, da, 1)
    b, rb = _tri(6, dtype, db, 2)
    got = _BIN[op](a, b)
    assert type(got).__name__ == "TriangularMatrix" and got.has_diagonal == want_diag
    np.testing.assert_array_equal(to_np(got), _BIN[op](ra, rb))


def test_F099_upper_plus_diagonal_and_upper_times_dense():
    a, ra = _tri(5, "float64", False, 3)
    d = pc.diagonal(np.arange(1.0, 6.0))
    s = a + d
    assert type(s).__name__ == "TriangularMatrix" and s.has_diagonal
    np.testing.assert_array_equal(to_np(s), ra + np.diag(np.arange(1.0, 6.0)))
    full = np.arange(25.0).reshape(5, 5)
    p = a * pc.matrix(full)
    assert type(p).__name__ == "TriangularMatrix" and not p.has_diagonal  # strict * dense
    np.testing.assert_array_equal(to_np(p), ra * full)
    t, rt = _tri(5, "float64", True, 4)
    q = t * pc.matrix(full)
    assert q.has_diagonal
    np.testing.assert_array_equal(to_np(q), rt * full)
    np.testing.assert_array_equal(to_np(t + t.T), rt + rt.T)  # upper + lower: dense


@pytest.mark.parametrize("dtype", ["float64", "int32", "bool", "complex_float64"])
def test_F112_integer_index_drops_the_axis_as_in_numpy(dtype):
    """`M[0, :]`, `M[:, 0]`, mixed integer / array and paired index arrays are 1-D in NumPy;
    PyCauset returned 1 x n / n x 1 matrices."""
    a = (np.arange(12).reshape(3, 4) % 3).astype({"complex_float64": "complex128"}.get(dtype, dtype))
    m = pc.matrix(a)
    keys = [(0, slice(None)), (slice(None), 2), (-1, slice(None, None, -2)), (np.array([0, 2]), 1),
            (1, np.array([3, 0])), (np.array([0, 1, 2]), np.array([3, 2, 1])), (slice(0, 1), slice(None))]
    for key in keys:
        got = m[key]
        assert got.shape == a[key].shape, key
        np.testing.assert_array_equal(np.asarray(got), a[key])


def test_F113_complex_infinities_survive_views_and_export():
    """Writing out a view multiplied every element by 1 + 0j, and (inf + 0j)(1 + 0j) is
    inf + nan j: a transposed complex matrix exported inf as inf + nan j, a scaled one twice
    multiplied (nan + nan j where NumPy gives inf + nan j)."""
    a = np.array([[np.inf, 1j], [-np.inf + 2j, 3]], dtype=complex)
    m = pc.matrix(a)
    np.testing.assert_array_equal(np.asarray(m.T), a.T)
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(np.asarray(m * 2), a * 2)
        b = np.array([[3 + 4j, 1], [1, 1]], dtype=complex)
        np.testing.assert_array_equal(np.asarray(m.T / pc.matrix(b).T), a.T / b.T)


def test_bulk_reader_windows_and_transposes_match_numpy():
    """Structured and transposed operands are written out through the bulk stored-value
    reader (math/RawRows.hpp): dense windows, transposed windows and transposes read the
    same elements as the per-element accessor did."""
    rng = np.random.default_rng(1)
    a = rng.integers(-50, 50, (40, 30)).astype(np.int32)
    m = pc.matrix(a)
    for got, want in [(m[3:20, 2:25], a[3:20, 2:25]), (m[3:20, 2:25].T, a[3:20, 2:25].T), (m.T, a.T)]:
        np.testing.assert_array_equal(np.asarray(~got), ~want)
        np.testing.assert_array_equal(np.asarray(got & got), want & want)
        np.testing.assert_array_equal(np.asarray(got * 3), want * 3)
        np.testing.assert_allclose(np.asarray(got / (got + 100)), want / (want + 100))
        np.testing.assert_allclose(np.asarray(got + 0.5), want + 0.5)


def test_unit_vector_dot_is_exact_and_keeps_numpy_nan_rules():
    """dot with a unit vector reads u_k y_k plus the 0 * y_i terms only for inf / NaN (it read
    every element of both through the virtual accessor, 0.05x NumPy). Integer dtypes give an
    exact int, as every other integer dot (they returned a float)."""
    n = 1000
    y = np.arange(n, dtype=np.int64) * 3 - 7
    u = pc.UnitVector(n, 5, dtype="int64")
    got = u.dot(pc.vector(y))
    assert type(got) is int and got == int(y[5])
    assert (u * 3).dot(pc.vector(y)) == 3 * int(y[5])
    for i, v in [(5, np.inf), (7, np.inf), (7, np.nan), (5, -np.inf)]:
        yf = np.ones(n)
        yf[i] = v
        with np.errstate(all="ignore"):
            want = np.dot(np.asarray(pc.UnitVector(n, 5)), yf)
        got = pc.UnitVector(n, 5).dot(pc.vector(yf))
        assert (np.isnan(got) and np.isnan(want)) or got == want, (i, v, got, want)


def test_bit_bulk_reader_matches_numpy():
    """Bit matrices (dense, causal, their transposes) are unpacked in bulk for whole-matrix
    kernels (math/RawRows.hpp): scalar ops, mixed-type element-wise ops and the norm read
    the same bits as the per-element accessor did."""
    rng = np.random.default_rng(3)
    ref = rng.random((37, 37)) < 0.4
    dense, causal = pc.matrix(ref), pc.causal_matrix(np.triu(ref, 1))
    for m in (dense, dense.T, causal, causal.T):
        a = np.asarray(m)
        np.testing.assert_array_equal(np.asarray(m + 3), a + 3)
        np.testing.assert_array_equal(np.asarray(m * 2.5), a * 2.5)
        np.testing.assert_allclose(np.asarray(m / (m + 1)), a / (a + 1.0))
        assert pc.norm(m) == pytest.approx(np.linalg.norm(a))


def test_antisymmetric_diagonal_reads_plus_zero_in_bulk_reads():
    """A materialized `anti * -1.0` stores 0 * -1 = -0.0 on the diagonal; the logical diagonal
    is +0.0 (F-078), so 1 / A is +inf there. The bulk reader must not pass the stored sign."""
    x = np.array([[0, 1.0, 2], [-1, 0, 3], [-2, -3, 0]])
    a = pc.antisymmetric(x) * -1.0
    d = pc.matrix(np.ones((3, 3)))
    with np.errstate(divide="ignore"):
        got = np.asarray(d / a)
    assert np.all(np.diag(got) == np.inf)
    assert np.all(np.diag(np.asarray(1.0 / a)) == np.inf)


def test_unit_vector_pairs_dot_exactly():
    """Two unit vectors: one product or none, in the right type (int for integers)."""
    n = 1000
    for dt in ("float64", "int16", "complex_float64"):
        u, v, w = pc.UnitVector(n, 5, dtype=dt), pc.UnitVector(n, 5, dtype=dt), pc.UnitVector(n, 7, dtype=dt)
        assert u.dot(v) == np.dot(np.asarray(u), np.asarray(v))
        assert u.dot(w) == 0
    assert type(pc.UnitVector(n, 1, dtype="int32").dot(pc.UnitVector(n, 1, dtype="int32"))) is int
    assert (pc.UnitVector(n, 1, dtype="int32") * 3).dot(pc.UnitVector(n, 1, dtype="int32") * -2) == -6


def test_float_and_complex_vector_views_stay_views(tmp_path):
    """-v, 2.5 * v and v.conj() of float / complex vectors are O(1) views (they were written
    out: complex -v took 12 ms for 1M elements); reads, writes, save / load see the view."""
    z = np.array([1 + 2j, -0.0 + 0j, 3 - 1j])
    v = pc.vector(z)
    neg = -v
    np.testing.assert_array_equal(np.asarray(neg), -z)
    np.testing.assert_array_equal(np.asarray(v * 2.5), z * 2.5)
    np.testing.assert_array_equal(np.asarray(v.conj()), z.conj())
    assert neg.dot(v) == pytest.approx(np.dot(-z, z))
    pc.save(neg, tmp_path / "n.pycauset")
    np.testing.assert_array_equal(np.asarray(pc.load(tmp_path / "n.pycauset")), -z)
    neg[0] = 5.0  # a write folds the view into the writer's own copy
    np.testing.assert_array_equal(np.asarray(neg), np.array([5.0, *(-z[1:])]))
    np.testing.assert_array_equal(np.asarray(v), z)
    f = pc.vector(np.array([0.0, -0.0, 1.5]))
    np.testing.assert_array_equal(np.signbit(np.asarray(-f)), np.signbit(-np.array([0.0, -0.0, 1.5])))


def test_complex32_bulk_reads_match_numpy():
    """complex_float16 matrices (two float16 planes) are read in bulk by whole-matrix kernels;
    values match NumPy's complex64 on data exact in float16."""
    a = ((np.arange(36).reshape(6, 6) % 7) / 4 + 1j * ((np.arange(36).reshape(6, 6) % 5) / 8)).astype(np.complex64)
    m = pc.matrix(a, dtype="complex_float16")
    for got, want in [(m, a), (m.T, a.T)]:
        np.testing.assert_allclose(np.asarray(got + (1 + 1j)), want + (1 + 1j), rtol=1e-3)
        np.testing.assert_allclose(np.asarray(got * 2), want * 2, rtol=1e-3)
        assert complex(pc.sum(got)) == pytest.approx(complex(want.sum()), rel=1e-3)
        np.testing.assert_allclose(np.asarray(got + got), want + want, rtol=1e-3)


def test_F117_written_closure_is_not_answered_from_shared_cache():
    """A write to C + C.T materializes the closure, but its storage is still C's bits (shared
    by C's other closures): the derived-value cache must not answer for it (it returned the
    pre-write sum, and another closure of C then got the written one's)."""
    c = np.triu(np.ones((4, 4), dtype=bool), 1)
    cm = pc.causal_matrix(c)
    s = cm + cm.T
    assert pc.sum(s) == 12
    s[0, 1] = 7
    assert pc.sum(s) == int(np.asarray(s).sum()) == 24
    assert pc.sum(cm + cm.T) == 12


def test_F118_bool_vector_write_copies_on_write():
    v = pc.vector(np.array([True, True, False]))
    assert pc.sum(v) == 2
    w = v.T
    w[0] = False
    np.testing.assert_array_equal(np.asarray(v), [True, True, False])
    assert pc.sum(v) == 2 and pc.sum(w) == 1


@pytest.mark.parametrize("view", ["scaled", "T", "int_scaled", "conj", "H_scaled"])
def test_F119_indexing_copies_keep_the_view(view):
    """Step slices, index arrays and the 1-D rows / columns of F-112 copied stored values and
    dropped the factor and conjugation (B = A * 2: B[0, :] was A[0, :])."""
    a = np.arange(16.0).reshape(4, 4)
    z = a + 1j * a.T
    m, r = {
        "scaled": (pc.matrix(a) * 2.0, a * 2),
        "T": (pc.matrix(a).T, a.T),
        "int_scaled": (pc.matrix((a * 10).astype(np.int8)) * 3, (a * 10).astype(np.int8) * np.int8(3)),
        "conj": (pc.matrix(z).conj(), z.conj()),
        "H_scaled": (pc.matrix(z).conj().T * 2.0, z.conj().T * 2),
    }[view]
    for key in [(0, slice(None)), (slice(None), 1), ([0, 1], [2, 3]), (slice(None, None, 2), slice(None, None, 2)),
                ([0, 2], slice(None))]:
        np.testing.assert_array_equal(np.asarray(m[key]), r[key])


# --- area 3.2 review, medium and low items (F-126) -----------------------------------------

_H = np.array([[1.0, 2 + 3j], [2 - 3j, 4.0]])


@pytest.mark.parametrize("c", [1j, 2 + 1j, -0.5j])
def test_F126_hermitian_times_non_real_drops_the_class(c):
    """C-4: H * 1j kept the Hermitian class with non-Hermitian values."""
    r = pc.hermitian(_H) * c
    assert not isinstance(r, (pc.HermitianMatrix, pc.SkewHermitianMatrix))
    np.testing.assert_allclose(to_np(r), _H * c)
    assert isinstance(pc.hermitian(_H) * 2.0, pc.HermitianMatrix)  # real factors keep it
    np.testing.assert_allclose(to_np(pc.hermitian(_H) / 1j), _H / 1j)


@pytest.mark.parametrize("make, value", [
    (lambda: pc.matrix(np.arange(4.0).reshape(2, 2)) * 2.0, 1.0),
    (lambda: pc.matrix(np.arange(4, dtype=np.int16).reshape(2, 2)) * 3, 5),
    (lambda: pc.matrix(np.ones((2, 2)) * (1 + 2j)).conj(), 1j),
    (lambda: pc.matrix(np.arange(9.0).reshape(3, 3))[0:2, 1:3] * 2.0, 7.0),
])
def test_F126_fill_sets_every_visible_element(make, value):
    """C-5: fill wrote raw values under the view, so (A * 2).fill(1) read 2."""
    m = make()
    m.fill(value)
    np.testing.assert_array_equal(to_np(m), np.full(m.shape, value))


def test_F126_set_identity_of_a_window():
    a = pc.zeros((4, 4))
    w = a[0:2, 1:3]
    w.set_identity()
    np.testing.assert_array_equal(to_np(w), np.eye(2))
    with pytest.raises(ValueError):
        a[0:2, 0:3].set_identity()


def _upper3(dtype="float64"):
    t = pc.TriangularMatrix(3, dtype=dtype, has_diagonal=True)
    t[0, 0], t[0, 1], t[1, 2], t[2, 2] = 1, 2, 3, 4
    return t


@pytest.mark.parametrize("cls, f", [(pc.SymmetricMatrix, lambda u: u + np.triu(u, 1).T),
                                    (pc.AntiSymmetricMatrix, lambda u: np.triu(u, 1) - np.triu(u, 1).T)])
def test_F126_from_triangular_reads_the_view(cls, f):
    """C-6: from_triangular copied the stored triangle, ignoring T * 2 and T.conj()."""
    t = _upper3()
    np.testing.assert_array_equal(to_np(cls.from_triangular(t * 2.0)), f(to_np(t) * 2.0))
    with pytest.raises(ValueError, match="upper"):
        cls.from_triangular(t.T)


def test_F126_hermitian_from_a_conjugated_triangle():
    t = pc.TriangularMatrix(2, dtype="complex_float64", has_diagonal=True)
    t[0, 0], t[0, 1], t[1, 1] = 1, 2 + 3j, 4
    u = np.conj(to_np(t))
    np.testing.assert_array_equal(to_np(pc.HermitianMatrix.from_triangular(t.conj())), np.triu(u) + np.triu(u, 1).conj().T)


@pytest.mark.parametrize("view", [lambda h: h * 2.0, lambda h: h.T, lambda h: h.conj()])
def test_F126_writes_into_hermitian_views_keep_the_view(view):
    """C-7: the write fold had no Hermitian case: the factor or transpose was dropped."""
    r = view(pc.hermitian(_H))
    expected = to_np(r).copy()
    r[0, 1] = 1 + 1j
    expected[0, 1], expected[1, 0] = 1 + 1j, 1 - 1j
    np.testing.assert_allclose(to_np(r), expected)


@pytest.mark.parametrize("make", [
    lambda: pc.symmetric(np.array([[1, 2j], [2j, 3]])) * 2.0,
    lambda: pc.diagonal(np.array([1 + 1j, 2])) * 2.0,
])
def test_F126_writes_into_complex_structured_views(make):
    r = make()
    expected = to_np(r).copy()
    r[1, 1] = 5j
    expected[1, 1] = 5j
    np.testing.assert_allclose(to_np(r), expected)


def test_F126_structured_reads_out_of_range_raise():
    """C-8: triangular get returned 0 for an out-of-range row."""
    for t in (pc.TriangularMatrix(4), pc.TriangularMatrix(4, dtype="bool")):
        with pytest.raises(IndexError):
            t.get_element_as_double(100, 0)


def test_F126_element_accessors_use_the_view_rule():
    """C-9: a complex multiply in DenseMatrix's own rule gave inf + nan j."""
    m = pc.matrix(np.array([[np.inf, -0.0], [1.0, 2.0]])) * 2.0
    assert m.get_element_as_complex(0, 0) == complex(np.inf, 0.0)
    b = pc.matrix(np.array([[True, False]])) * -1.0
    assert np.signbit(b.get_element_as_double(0, 1))


def test_F126_random_bit_matrices_check_p_and_keep_the_seed():
    """C-11."""
    for cls in (pc.TriangularMatrix, pc.DenseMatrix):
        for p in (1.5, -0.1, float("nan")):
            with pytest.raises(ValueError):
                cls.random(5, p=p)
    assert pc.TriangularMatrix.random(5, seed=3).seed == 3
    a = to_np(pc.TriangularMatrix.random(300, p=0.5, seed=2))
    b = to_np(pc.TriangularMatrix.random(300, p=0.50005, seed=2))
    assert not np.array_equal(a, b)


def test_F126_zero_outside_the_triangle_is_accepted():
    """C-12: T[1, 0] = 0.0 raised for float triangles (bit and diagonal accepted it)."""
    t = _upper3()
    t[1, 0] = 0.0
    t[:, :] = 0.0
    np.testing.assert_array_equal(to_np(t), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        t[1, 0] = 1.0


@pytest.mark.parametrize("make", [
    lambda: pc.vector(np.array([1, 2], np.int32)).conj(),
    lambda: pc.vector(np.array([True, False])).conj(),
    lambda: pc.matrix(np.array([[True, False], [False, True]])).conj(),
])
def test_F126_write_to_conj_of_real_values(make):
    """C-13: writing to conj() of an int / bool object raised ("carries a scale factor")."""
    m = make()
    idx = 1 if len(m.shape) == 1 else (0, 1)
    m[idx] = 1
    assert to_np(m)[idx] == 1


# --- area 3.3 review (F-130) ---------------------------------------------------------------

@pytest.mark.parametrize("values", [
    [np.complex64(1j), 2], [np.int64(2**53 + 1)], [np.int8(1)], [np.uint64(2**64 - 1)],
    [np.float16(0.5), np.float16(1)], [np.bool_(True), np.bool_(False)], [np.float32(1), 2.5],
])
def test_F130_lists_with_numpy_scalars_infer_like_numpy(values):
    """C-5: list input tested only for Python types: np.complex64(1j) lost its imaginary part,
    np.int64 values were rounded through float64."""
    want = np.array(values)
    v = pc.vector(values)
    assert np.asarray(v).dtype == want.dtype
    np.testing.assert_array_equal(np.asarray(v), want)
    m = pc.matrix([values])
    assert np.asarray(m).dtype == want.dtype
    np.testing.assert_array_equal(np.asarray(m)[0], want)


def test_F130_empty_vector_is_float64():
    assert pc.vector([]).dtype == "float64"


@pytest.mark.parametrize("token, want", [("i8", "int64"), ("u8", "uint64"), ("f4", "float32"), ("f", "float32"),
                                         ("<f8", "float64"), ("F", "complex_float32"), ("int", "int32"),
                                         ("bit", "bool"), ("single", "float32"), ("INT16", "int16")])
def test_F133_dtype_tokens_mean_what_numpy_says(token, want):
    """3.3 C-9: "i8" gave int8 (NumPy: int64), "u8" uint8; NumPy codes like "f4" raised."""
    assert pc.zeros((1, 1), dtype=token).dtype == want


@pytest.mark.parametrize("token", [np.longdouble, np.clongdouble, "f16", "float_"])
def test_F133_unstored_dtypes_raise(token):
    with pytest.raises((TypeError, ValueError)):
        pc.zeros((1, 1), dtype=token)


@pytest.mark.parametrize("make_a, make_b", [
    (lambda: pc.diagonal(np.array([1.0, 2.0, 3.0])), lambda: pc.diagonal(np.array([5.0]))),
    (lambda: pc.diagonal(np.array([1.0, 2.0, 3.0])), lambda: pc.matrix(np.array([[5.0]]))),
    (lambda: pc.symmetric(np.array([[1.0, 2.0], [2.0, 3.0]])), lambda: pc.diagonal(np.array([4.0]))),
    (lambda: pc.hermitian(np.array([[1.0, 2j], [-2j, 3.0]])), lambda: pc.hermitian(np.array([[4.0]]))),
    (lambda: pc.identity(3), lambda: pc.identity(1)),
])
@pytest.mark.parametrize("op", ["+", "-", "*"])
def test_F130_structured_operands_broadcast_like_numpy(make_a, make_b, op):
    """C-1: a 1 x 1 operand is square too: Diagonal(3) + Diagonal(1) came back diagonal without
    the off-diagonal values; Identity(3) + Identity(1) raised."""
    a, b = make_a(), make_b()
    for x, y in ((a, b), (b, a)):
        np.testing.assert_allclose(to_np(_BIN[op](x, y)), _BIN[op](to_np(x), to_np(y)))


@pytest.mark.parametrize("dtype, big", [("int64", 2**60 + 1), ("uint64", 2**64 - 1)])
def test_F130_identity_matmul_is_exact_for_wide_integers(dtype, big):
    """C-2: I @ X multiplied by the Python float 1.0, through float64."""
    x = pc.matrix(np.array([[big, 1], [2, 3]], dtype=dtype))
    i = pc.IdentityMatrix(2, dtype=dtype)
    for r in (i @ x, x @ i):
        assert str(r.dtype) == dtype
        assert int(to_np(r)[0, 0]) == big


def test_F130_numpy_float64_scalars_are_numpy_scalars():
    """C-4: np.float64 / np.complex128 are subclasses of Python float / complex and were taken
    as weak Python scalars."""
    m = pc.matrix(np.ones((2, 2), dtype=np.float32))
    i = pc.matrix(np.ones((2, 2), dtype=np.int8))
    # A NumPy scalar is an operand of its dtype (DType System §9): under the matrix rules a
    # scalar never widens a float matrix, and np.float64 with int8 is float64.
    assert str((m + np.float64(1)).dtype) == "float32"
    assert str((m * np.complex128(1j)).dtype) == "complex_float32"
    assert str((i + np.float64(1)).dtype) == "float64"
    assert str((i + np.float32(1)).dtype) == "float32"


def test_F130_signed_uint64_error_names_the_dtypes():
    with pytest.raises(ValueError, match="int8 and uint64"):
        pc.matrix(np.ones((2, 2), np.int8)) + pc.matrix(np.ones((2, 2), np.uint64))


def test_size_is_an_attribute_as_in_numpy():
    assert pc.zeros((2, 3)).size == 6
    assert pc.vector(np.arange(4.0)).size == 4


def test_F129_thunk_blocks_notice_writes_to_native_leaves():
    """3.9 C-4: ThunkBlock read a `version` attribute native matrices do not have, so a block
    product kept (and saved) its old value after a leaf was written."""
    from pycauset._internal.thunks import StaleThunkError, ThunkBlock

    a = pc.matrix(np.ones((2, 2)))
    t = ThunkBlock(rows=2, cols=2, compute=lambda: a @ a, sources_for_staleness=[a])
    assert float(to_np(t.materialize())[0, 0]) == 2.0
    a[0, 0] = 7.0
    with pytest.raises(StaleThunkError):
        t.materialize()
    v = a.T  # a view shares the storage; a write copies it first, with a fresh version
    t2 = ThunkBlock(rows=2, cols=2, compute=lambda: v @ v, sources_for_staleness=[v])
    t2.materialize()
    v[0, 1] = 3.0
    with pytest.raises(StaleThunkError):
        t2.materialize()


@pytest.mark.parametrize("cls, args", [(pc.DenseMatrix, (2,)), (pc.TriangularMatrix, (3,)), (pc.DiagonalMatrix, (2,)),
                                       (pc.SymmetricMatrix, (2,)), (pc.IdentityMatrix, (2,)), (pc.DenseVector, (3,))])
@pytest.mark.parametrize("token, want", [("bit", "bool"), (np.float32, "float32"), ("i8", "int64"),
                                         (np.dtype("int16"), "int16"), ("single", "float32"), (int, "int32")])
def test_F133_class_constructors_take_every_dtype_token(cls, args, token, want):
    """3.3 C-9: the native constructors parsed dtype names with their own table:
    pc.DenseMatrix(2, dtype="bit") raised while pc.zeros accepted it."""
    assert str(cls(*args, dtype=token).dtype) == want


def test_F133_class_constructors_reject_unstored_dtypes():
    with pytest.raises(TypeError, match="unsupported dtype"):
        pc.DenseMatrix(2, dtype=np.longdouble)


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint64", "float16", "float64", "complex_float16", "complex128"])
@pytest.mark.parametrize("storage", ["ram", "file"])
def test_zeros_needs_no_fill(dtype, storage):
    """pc.zeros relies on new storage being zero (it filled every byte again: 18 ms for
    4000 x 4000, NumPy 0). RAM is calloc'd; a new file is sized, not written."""
    previous = pc.get_memory_threshold()
    if storage == "file":
        pc.set_memory_threshold(1)
    try:
        for shape in ((70, 3), (5,)):
            z = pc.zeros(shape, dtype=dtype)
            assert not np.any(to_np(z))
    finally:
        pc.set_memory_threshold(previous)


# --- area 3.6 review (F-136) ---------------------------------------------------------------

@pytest.mark.parametrize("a, b, dtype", [([[2**31]], [[2**32]], "int64"), ([[2**62]], [[2]], "int64"),
                                         ([[-(2**62)]], [[2]], "int64")])
def test_F136_integer_matmul_overflow_raises(a, b, dtype):
    """C-1: products multiplied in unchecked int64 wrapped (2**31 * 2**32 gave -2**63)."""
    x, y = pc.matrix(np.array(a, dtype=dtype)), pc.matrix(np.array(b, dtype=dtype))
    exact = int(a[0][0]) * int(b[0][0])
    if -(2**63) <= exact < 2**63:
        assert int(to_np(x @ y)[0, 0]) == exact
    else:
        with pytest.raises(OverflowError):
            x @ y
    d = pc.diagonal(np.array([a[0][0], 1], dtype=dtype))
    e = pc.diagonal(np.array([b[0][0], 1], dtype=dtype))
    if -(2**63) <= exact < 2**63:
        assert int(to_np(d @ e)[0, 0]) == exact
    else:
        with pytest.raises(OverflowError):
            d @ e


def test_F136_uint64_matmul_above_2_63_is_exact():
    """C-1: uint64 values >= 2**63 were read as negative and raised although the result fits."""
    x = pc.matrix(np.array([[2**63, 0], [0, 1]], dtype=np.uint64))
    y = pc.matrix(np.array([[1, 0], [0, 1]], dtype=np.uint64))
    assert int(to_np(x @ y)[0, 0]) == 2**63
    t = pc.TriangularMatrix(2, dtype="uint64", has_diagonal=True)
    t[0, 0], t[1, 1] = 2**63, 1
    u = pc.TriangularMatrix(2, dtype="uint64", has_diagonal=True)
    u[0, 0], u[1, 1] = 1, 1
    assert int(to_np(t @ u)[0, 0]) == 2**63


@pytest.mark.parametrize("op", ["det", "slogdet", "inv", "solve", "qr", "svd", "eigvalsh", "pinv"])
def test_F136_nan_input_gives_numpy_results_not_errors(op):
    """C-2: LAPACKE's NaN check rejected every factorisation of NaN input ("illegal argument")."""
    a = np.array([[1.0, np.nan], [np.nan, 2.0]])
    m = pc.matrix(a)
    fns = {"det": (pc.determinant, np.linalg.det), "slogdet": (pc.slogdet, np.linalg.slogdet),
           "inv": (pc.invert, np.linalg.inv), "solve": (lambda x: pc.solve(x, pc.matrix(np.eye(2))),
                                                        lambda x: np.linalg.solve(x, np.eye(2))),
           "qr": (lambda x: pc.qr(x)[1], lambda x: np.linalg.qr(x)[1]),
           "svd": (pc.svdvals if hasattr(pc, "svdvals") else (lambda x: pc.svd(x, compute_uv=False)),
                   lambda x: np.linalg.svd(x, compute_uv=False)),
           "eigvalsh": (pc.eigvalsh, np.linalg.eigvalsh), "pinv": (pc.pinv, np.linalg.pinv)}
    ours, theirs = fns[op]
    try:
        want = theirs(a)
    except np.linalg.LinAlgError:
        with pytest.raises((RuntimeError, ValueError)):
            ours(m)
        return
    got = ours(m)
    def arr(v):
        if isinstance(v, tuple):
            return np.concatenate([np.ravel(np.asarray(x, dtype=complex)) for x in v])
        return np.ravel(np.asarray(v if not hasattr(v, "shape") or isinstance(v, np.ndarray) else to_np(v), dtype=complex))
    assert np.array_equal(np.isnan(arr(got)), np.isnan(arr(want)))


@pytest.mark.parametrize("op", ["cholesky", "eigh", "eigvalsh", "svd", "pinv"])
def test_F136_empty_matrices(op):
    """C-2: 0 x 0 input passed lda = 0, which LAPACK rejects."""
    f = getattr(pc, op)
    f(pc.matrix(np.zeros((0, 0))))


def test_F136_rectangular_identity_product():
    """C-3: IdentityMatrix(3, 2) @ IdentityMatrix(2, 3) returned the full 3 x 3 identity."""
    r = pc.IdentityMatrix(3, 2) @ pc.IdentityMatrix(2, 3)
    np.testing.assert_array_equal(to_np(r), np.eye(3, 2) @ np.eye(2, 3))
    np.testing.assert_array_equal(to_np(pc.IdentityMatrix(2, 3) @ pc.IdentityMatrix(3, 2)), np.eye(2))


def test_F136_real_factors_stay_real():
    """C-4: (1 + 0j) * (inf + 0j) is inf + nan j: `A * inf` of a real matrix got a complex
    factor; sum returned a complex number and det raised."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = pc.matrix(a) * np.inf
    assert m.dtype == "float64" and not isinstance(pc.sum(m), complex)
    np.testing.assert_array_equal(to_np(m), a * np.inf)
    s = pc.matrix(a) * 2.0 * np.inf
    assert complex(s.scalar).imag == 0.0


def test_F136_division_by_a_subnormal():
    """C-4: A / s was A * (1/s); 1/1e-310 overflows: [1e-10, 0] / 1e-310 gave [inf, nan]."""
    a = np.array([[1e-10, 0.0]])
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(to_np(pc.matrix(a) / 1e-310), a / 1e-310)
    np.testing.assert_array_equal(to_np(pc.matrix(a) / 4.0), a / 4.0)  # still an O(1) view


def test_F136_sum_of_a_scaled_view_scales_each_element():
    """C-5: the factor was applied after summing: sum([1e308, -1e308] * 10) gave 0, NumPy nan."""
    a = np.array([[1e308, -1e308]])
    with np.errstate(over="ignore", invalid="ignore"):
        want = np.sum(a * 10)
    got = pc.sum(pc.matrix(a) * 10.0)
    assert np.isnan(got) and np.isnan(want)
    b = np.arange(12.0).reshape(3, 4)
    assert pc.sum(pc.matrix(b) * 0.5) == pytest.approx(np.sum(b * 0.5))


def test_3_6_C6_integer_conj_sums_exactly():
    """Review 3.6 C-6: integer .conj() / .H views summed and traced in float."""
    a = np.array([[2**60 + 1, 1], [3, 2**60 + 1]], np.int64)
    A = pc.matrix(a)
    for v in (A.conj(), A.H):
        assert pc.sum(v) == int(a.sum()) and isinstance(pc.sum(v), int)
        assert pc.trace(v) == int(np.trace(a))


def test_3_6_C10_skew_eigen_entries_check_their_input():
    """Review 3.6 C-10: eigvals_skew read one triangle of any input (a symmetric matrix gave
    imaginary eigenvalues), complex input was dropped or failed inside, 0 x 0 raised."""
    rng = np.random.default_rng(0)
    s = rng.standard_normal((6, 6))
    sym = pc.matrix(s + s.T)
    with pytest.raises(ValueError, match="skew-symmetric"):
        pc.eigvals_skew(sym, 3)
    with pytest.raises(ValueError, match="skew-symmetric"):
        pc.eig_skew(sym, 2)
    k = s - s.T
    with pytest.raises(TypeError, match="real matrices only"):
        pc.eigvals_skew(pc.matrix(1j * k), 3)
    with pytest.raises(TypeError, match="real matrices only"):
        pc.eigvals_arnoldi(pc.matrix(1j * k), 2, 4)
    w = np.asarray(pc.eigvals_skew(pc.matrix(k), 6))
    want = np.linalg.eigvals(k)
    assert np.allclose(np.sort(np.abs(w.imag)), np.sort(np.abs(want.imag)))
    assert np.asarray(pc.eigvals_skew(pc.matrix(np.zeros((0, 0))), 1)).shape == (0,)
    w0, v0 = pc.eig_skew(pc.matrix(np.zeros((0, 0))), 1)
    assert np.asarray(w0).shape == (0,) and np.asarray(v0).shape == (0, 0)


def test_3_6_C11_materialize_keeps_signed_zeros():
    """Review 3.6 C-11: materialize added 0, and -0.0 + 0 is +0.0."""
    a = np.array([[-0.0, 1.0], [0.0, -2.0]])
    got = np.asarray(pc.materialize(-pc.matrix(a)))
    assert np.array_equal(np.signbit(got), np.signbit(-a))


def test_3_6_C13_bool_scalar_is_a_bool_operand():
    """Review 3.6 C-13: a Python bool / np.bool_ was an int, so B * True was int32 (NumPy bool)."""
    b = np.array([[True, False], [False, True]])
    B = pc.matrix(b)
    for s in (True, False, np.True_, np.False_):
        got = B * s
        assert got.dtype == "bool"
        assert np.array_equal(np.asarray(got), b * s)
        assert np.array_equal(np.asarray(s * B), b * s)
    assert np.array_equal(np.asarray(B.T * True), b.T)
    # Integer and float matrices keep their dtype, as in NumPy.
    i = pc.matrix(np.array([[1, 2]], np.int8))
    assert (i * True).dtype == "int8" and (i + np.True_).dtype == "int8"
    f = pc.matrix(np.array([[1.5]], np.float32))
    assert (f * True).dtype == "float32"


def test_singular_matrix_is_a_value_error():
    """NumPy raises LinAlgError (a ValueError) for a singular matrix; PyCauset raised RuntimeError."""
    z = pc.matrix(np.zeros((2, 2)))
    for f in (pc.invert, lambda m: pc.matrix_power(m, -1), lambda m: pc.solve(m, pc.matrix(np.ones((2, 1))))):
        with pytest.raises(ValueError, match="singular"):
            f(z)
    d = pc.DiagonalMatrix(2, dtype="float64")  # zero diagonal: the structure rule
    with pytest.raises(ValueError, match="singular"):
        pc.invert(d)
    with pytest.raises(np.linalg.LinAlgError):  # NumPy's own class
        pc.invert(z)
    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        pc.cholesky(pc.matrix(-np.eye(2)))
