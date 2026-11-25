"""Shared fixtures for the new test suite (see review/TESTS.md).

Tests import pycauset from the source tree: pyproject sets `pythonpath = ["python"]`,
and the compiled extension must sit next to the sources. No sys.path manipulation in
tests (see review/TESTS.md).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import pycauset as pc

# PYCAUSET_LAZY=0 runs the whole suite with lazy results (views, closures) disabled: the
# eager paths must be correct on their own (LAZY.md section 10, the lazy == eager oracle).
LAZY = os.environ.get("PYCAUSET_LAZY", "1") != "0"
pc._native._set_lazy(LAZY)
requires_lazy = pytest.mark.skipif(not LAZY, reason="asserts a lazy result; PYCAUSET_LAZY=0")


def to_np(x):
    """Export any pycauset object (or scalar) to NumPy, bypassing the size guard."""
    if isinstance(x, (bool, int, float, complex, np.generic, np.ndarray)):
        return np.asarray(x)
    return pc.to_numpy(x, allow_huge=True)


@pytest.fixture
def force_disk():
    """Make every new matrix file-backed for the duration of a test."""
    previous = pc.get_memory_threshold()
    pc.set_memory_threshold(1024)
    yield
    pc.set_memory_threshold(previous if previous is not None else 8 * 1024**3)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
