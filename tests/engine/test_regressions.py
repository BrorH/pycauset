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
