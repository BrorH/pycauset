"""The out-of-core StreamingManager path (F-026): forced, traced, compared with NumPy.

CpuDevice switches to the tiled StreamingManager only when an operation doesn't fit the
memory governor's RAM budget (real free RAM minus a safety margin). Ordinary tests never
reach it, so it went untested. Here a huge safety margin makes every operation "not fit",
and the kernel trace confirms the streaming path actually ran.
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

native = pc._native


class _Streaming:
    """Context manager: inside it, no operation "fits in RAM", so ops stream."""

    def __enter__(self):
        self.previous = native._get_ram_safety_margin()
        native._debug_clear_kernel_trace()
        native._set_ram_safety_margin(1 << 62)
        return self

    def __exit__(self, *exc):
        native._set_ram_safety_margin(self.previous)
        return False


@pytest.fixture
def force_streaming():
    return _Streaming()


def _ran_streaming(kind: str) -> bool:
    return native._debug_last_kernel_trace() == f"streaming.{kind}"


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int64"])
@pytest.mark.parametrize("shape", [(64, 64), (97, 53)])
@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
def test_streaming_elementwise_matches_numpy(op, shape, dtype, force_streaming, rng):
    a = (rng.integers(1, 9, size=shape)).astype(dtype)
    b = (rng.integers(1, 9, size=shape)).astype(dtype)
    fn = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.true_divide}[op]
    x, y = pc.matrix(a), pc.matrix(b)  # operands are created with the normal budget
    with force_streaming:
        result = eval(f"x {op} y", {}, {"x": x, "y": y})
        # Integer true division yields float64 (mixed dtypes): not streamed, by design.
        if not (op == "/" and dtype.startswith("int")):
            assert _ran_streaming("elementwise"), native._debug_last_kernel_trace()
    got = to_np(result)
    expected = fn(a.astype(np.complex128), b.astype(np.complex128))
    np.testing.assert_allclose(got.astype(np.complex128), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("m,k,n", [(64, 64, 64), (130, 70, 90), (1, 200, 1), (300, 5, 300)])
def test_streaming_matmul_matches_numpy(m, k, n, dtype, force_streaming, rng):
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    x, y = pc.matrix(a), pc.matrix(b)
    with force_streaming:
        result = x @ y
        assert _ran_streaming("matmul"), native._debug_last_kernel_trace()
    got = to_np(result)
    tol = 1e-4 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(got, a.astype(np.float64) @ b.astype(np.float64), rtol=tol, atol=tol * k)


# F-040: the streaming engine reads raw storage, so views and unsupported dtypes must
# take the direct path (which honours view metadata) and still give correct results.
VIEW_CASES = {
    "A.T @ B": (lambda x, y: x.T @ y, lambda a, b: a.T @ b),
    "(A*2.5) @ B": (lambda x, y: (x * 2.5) @ y, lambda a, b: (a * 2.5) @ b),
    "A @ (B*-1.5)": (lambda x, y: x @ (y * -1.5), lambda a, b: a @ (b * -1.5)),
    "A.T + B": (lambda x, y: x.T + y, lambda a, b: a.T + b),
    "(A*2.5) - B": (lambda x, y: (x * 2.5) - y, lambda a, b: (a * 2.5) - b),
}


@pytest.mark.parametrize("case", list(VIEW_CASES))
def test_streaming_with_views_is_correct(case, force_streaming, rng):
    pc_fn, np_fn = VIEW_CASES[case]
    a = rng.standard_normal((70, 70))
    b = rng.standard_normal((70, 70))
    x, y = pc.matrix(a), pc.matrix(b)
    with force_streaming:
        result = pc_fn(x, y)
    np.testing.assert_allclose(to_np(result), np_fn(a, b), rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("dtypes", [("complex128", "complex128"), ("float64", "int32"), ("float32", "float64")])
@pytest.mark.parametrize("op", ["+", "*", "@"])
def test_streaming_unsupported_dtypes_fall_back_correctly(op, dtypes, force_streaming, rng):
    a = rng.integers(1, 5, size=(40, 40)).astype(dtypes[0])
    b = rng.integers(1, 5, size=(40, 40)).astype(dtypes[1])
    x, y = pc.matrix(a), pc.matrix(b)
    with force_streaming:
        result = eval(f"x {op} y", {}, {"x": x, "y": y})
    expected = eval(f"a {op} b", {}, {"a": a.astype(np.complex128), "b": b.astype(np.complex128)})
    np.testing.assert_allclose(to_np(result).astype(np.complex128), expected, rtol=1e-5, atol=1e-5)
