"""IO traces report facts about a call: its operands and the kernels the engine ran (F-100).

The trace used to carry a Python guess (a "route", tile shape and queue depth planned from a
threshold that the engine never read). Now it lists the kernels the engine ran.
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc

native = pc._native


@pytest.fixture(autouse=True)
def _clean():
    pc.clear_io_traces()
    yield
    pc.clear_io_traces()


def test_trace_records_operands_and_kernels():
    a = pc.matrix(np.arange(6.0).reshape(2, 3))
    b = pc.matrix(np.ones((3, 4)))
    pc.matmul(a, b)
    trace = pc.last_io_trace()
    assert trace["op"] == "matmul"
    assert [o["shape"] for o in trace["operands"]] == [(2, 3), (3, 4)]
    assert [o["dtype"] for o in trace["operands"]] == ["float64", "float64"]
    assert [o["storage"] for o in trace["operands"]] == ["memory", "memory"]
    assert trace["kernels"], "the engine logs at least one kernel for a matmul"


def test_trace_kernels_are_the_engines_log():
    a = pc.matrix(np.eye(3) * 2.0)
    native._debug_kernel_log_start()
    pc.determinant(a)
    outer = list(native._debug_kernel_log_stop())
    # The trace's own log nests inside the outer one and leaves it intact (DebugTrace.cpp).
    assert pc.last_io_trace("determinant")["kernels"] == outer


def test_traces_are_kept_per_op():
    a = pc.matrix(np.diag([1.0, 2.0]))
    pc.eigvalsh(a)
    pc.trace(a)
    assert pc.last_io_trace()["op"] == "trace"
    assert pc.last_io_trace("eigvalsh")["op"] == "eigvalsh"
    assert pc.last_io_trace("matmul") is None


def test_nested_traced_ops_record_only_the_outer_call():
    a = pc.matrix(np.array([[4.0, 1.0], [2.0, 3.0]]))
    pc.cond(a, 1)  # cond calls invert and norm, both traced on their own
    assert pc.last_io_trace()["op"] == "cond"
    assert pc.last_io_trace("inv") is None


def test_file_backed_operand_is_reported(tmp_path):
    path = tmp_path / "m.pycauset"
    pc.save(pc.matrix(np.eye(4)), str(path))
    m = pc.load(str(path))
    pc.trace(m)
    (snap,) = pc.last_io_trace()["operands"]
    assert snap["storage"] in ("file", "temp") and snap["backing_file"]


def test_the_threshold_knob_that_controlled_nothing_is_gone():
    assert not hasattr(pc, "set_io_streaming_threshold")
    assert not hasattr(pc, "get_io_streaming_threshold")
