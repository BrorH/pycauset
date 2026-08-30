"""Regression tests for the matrix() `storage=` kwarg.

`storage=` used to be silently dropped for NumPy input (every caller got an
anonymous `:memory:` matrix), which made the disk I/O and out-of-core
benchmarks benchmark RAM instead of disk. These tests pin the now-honoured
behaviour and the validation errors.
"""
import numpy as np
import pytest

import pycauset


def _backing(mat):
    return mat.get_backing_file()


def test_storage_default_is_ram():
    mat = pycauset.matrix(np.random.rand(32, 32))
    assert _backing(mat) == ":memory:"


def test_storage_ram_is_memory():
    mat = pycauset.matrix(np.random.rand(32, 32), storage="ram")
    assert _backing(mat) == ":memory:"


def test_storage_disk_is_file_backed_and_roundtrips():
    data = np.random.rand(48, 48)
    mat = pycauset.matrix(data, storage="disk")
    try:
        assert _backing(mat) != ":memory:"
        np.testing.assert_array_equal(pycauset.to_numpy(mat, allow_huge=True), data)
    finally:
        mat.close()


def test_storage_disk_for_different_dtypes():
    for dtype in (np.float32, np.float64, np.int32, np.bool_):
        data = np.random.rand(20, 20).astype(dtype) if np.issubdtype(dtype, np.floating) else np.random.randint(0, 2, size=(20, 20)).astype(dtype)
        mat = pycauset.matrix(data, storage="disk")
        try:
            assert _backing(mat) != ":memory:"
            np.testing.assert_array_equal(pycauset.to_numpy(mat, allow_huge=True), data)
        finally:
            mat.close()


def test_storage_invalid_value_raises():
    with pytest.raises(ValueError, match="storage must be 'ram' or 'disk'"):
        pycauset.matrix(np.random.rand(4, 4), storage="bogus")


def test_storage_disk_requires_numpy_input():
    with pytest.raises(TypeError, match="storage='disk' requires NumPy-array input"):
        pycauset.matrix([[1.0, 2.0], [3.0, 4.0]], storage="disk")
