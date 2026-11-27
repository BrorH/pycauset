"""Storage choices, the backing directory, export limits and container validation (TESTS.md
layout: test_storage.py). Moved from the old suite while retiring it."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import pycauset as pc


@pytest.fixture
def backing_dir(tmp_path):
    old = pc._storage_root()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pc.PyCausetStorageWarning)
        pc.set_backing_dir(tmp_path)
    yield tmp_path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pc.PyCausetStorageWarning)
        pc.set_backing_dir(old)


def test_storage_kwarg(backing_dir):
    assert pc.matrix(np.eye(3), storage="ram").get_backing_file() == ":memory:"
    disk = pc.matrix(np.eye(3), storage="disk")
    assert disk.get_backing_file().startswith(str(backing_dir))
    np.testing.assert_array_equal(pc.to_numpy(disk, allow_huge=True), np.eye(3))
    with pytest.raises(ValueError, match="'ram' or 'disk'"):
        pc.matrix(np.eye(2), storage="gpu")


def test_new_disk_allocations_follow_the_backing_dir(backing_dir, tmp_path_factory):
    previous = pc.get_memory_threshold()
    pc.set_memory_threshold(1)  # every new matrix is file-backed
    try:
        m1 = pc.TriangularMatrix(64, dtype="bool")
        assert Path(m1.get_backing_file()).resolve().parent == backing_dir.resolve()
        other = tmp_path_factory.mktemp("other")
        # Class constructors register the object; changing the directory under it warns.
        # (pc.matrix results are not registered: PLAN 4.3, the tracking is partial.)
        with pytest.warns(pc.PyCausetStorageWarning, match="Changing the backing directory"):
            pc.set_backing_dir(other)
        m2 = pc.TriangularMatrix(64, dtype="bool")
        assert Path(m2.get_backing_file()).resolve().parent == other.resolve()
        m1.close()
        m2.close()
    finally:
        pc.set_memory_threshold(previous)


def test_export_ceiling():
    pc.set_export_max_bytes(16)
    try:
        with pytest.raises(RuntimeError, match="materialization limit"):
            np.asarray(pc.matrix(np.eye(4)))
        np.testing.assert_array_equal(pc.to_numpy(pc.matrix(np.eye(4)), allow_huge=True), np.eye(4))
    finally:
        pc.set_export_max_bytes(None)


@pytest.mark.parametrize("payload", [b"PYC", b"not a container at all...." * 10])
def test_load_rejects_non_containers(tmp_path, payload):
    path = tmp_path / "x.pycauset"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="unsupported file format"):
        pc.load(str(path))


def test_load_rejects_unknown_format_version(tmp_path):
    path = tmp_path / "x.pycauset"
    pc.save(pc.matrix(np.eye(2)), str(path))
    data = bytearray(path.read_bytes())
    data[8] = 9  # version field after the 8-byte magic
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="unsupported format version: 9"):
        pc.load(str(path))


def test_load_rejects_a_truncated_header(tmp_path):
    path = tmp_path / "x.pycauset"
    pc.save(pc.matrix(np.eye(2)), str(path))
    path.write_bytes(path.read_bytes()[:12])  # magic, then less than the 16-byte preamble
    with pytest.raises(ValueError, match="truncated header|unsupported file format"):
        pc.load(str(path))
