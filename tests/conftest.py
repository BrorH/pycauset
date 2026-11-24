"""Shared fixtures for the new test suite (see review/TESTS.md).

Tests import pycauset from the source tree: pyproject sets `pythonpath = ["python"]`,
and the compiled extension must sit next to the sources. No sys.path manipulation in
tests (see review/TESTS.md).
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc


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
