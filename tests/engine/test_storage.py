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
        with pytest.raises(RuntimeError, match="ceiling"):
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


class _Closable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_runtime_cleanup_closes_tracked_objects_and_sweeps_tmp_files(tmp_path):
    """A private Runtime (the global one would close every live matrix of the test run)."""
    from pycauset import _storage
    from pycauset._internal.runtime import Runtime

    rt = Runtime(cleanup_storage=_storage.cleanup_storage, set_temporary_file=_storage.set_temporary_file)
    rt.set_storage_root(tmp_path)
    (tmp_path / "keep.pycauset").write_bytes(b"x")
    tmp = tmp_path / "spill_1.tmp"
    tmp.write_bytes(b"x")
    _storage.record_temporary_file(tmp)
    obj = _Closable()
    rt.track_matrix(obj)
    assert rt.has_live_matrices()

    rt.cleanup_all_roots()  # closes live objects, sweeps *.tmp only
    assert obj.closed
    assert not tmp.exists() and (tmp_path / "keep.pycauset").exists()


def test_export_size_estimate_for_objects_without_a_native_exporter():
    from pycauset._internal import export_guard

    class Fake:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype

    assert export_guard.estimate_materialized_bytes(Fake((3, 4), "float64")) == 96
    assert export_guard.estimate_materialized_bytes(Fake((5,), "int16")) == 10  # 1-D: n x 1
    assert export_guard.estimate_materialized_bytes(Fake((2, 2), "complex_float16")) == 32  # exported as complex64
    assert export_guard.estimate_materialized_bytes(Fake((2, 2), "BoolMatrix")) == 4  # class-style tokens
    assert export_guard.estimate_materialized_bytes(Fake((2, 2), "unknown")) is None


def test_storage_sweep_helpers(tmp_path):
    from pycauset import _storage

    _storage.cleanup_storage(tmp_path / "missing")  # a missing root is a no-op
    inside = tmp_path / "a.tmp"
    inside.write_bytes(b"x")
    _storage.record_temporary_file(inside)
    _storage.clear_tracked_temp_files(tmp_path)
    assert inside.exists()  # clearing the tracker never deletes
    _storage.cleanup_storage(tmp_path)
    assert not inside.exists()


# --- area 3.1 review: several processes, one storage directory (F-121) ---------------------

_CHILD = r"""
import sys, time
import numpy as np
import pycauset as pc
pc.set_backing_dir(sys.argv[1])
m = pc.matrix(np.full((32, 32), float(sys.argv[2])), storage="disk")
print(m.get_backing_file(), flush=True)
sys.stdin.readline()  # hold the file until the parent says go
print(float(np.asarray(m)[0, 0]), flush=True)
"""


def _child(backing_dir, value):
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=str(Path(pc.__file__).resolve().parents[1]))
    p = subprocess.Popen([sys.executable, "-W", "ignore", "-c", _CHILD, str(backing_dir), str(value)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    return p, p.stdout.readline().strip()


def test_F121_temp_names_carry_the_pid(backing_dir):
    import os

    m = pc.matrix(np.ones((4, 4)), storage="disk")
    assert f"_{os.getpid()}_" in Path(m.get_backing_file()).name


@pytest.mark.skipif(__import__("os").name == "nt", reason="liveness check is POSIX (Windows locks open files)")
def test_F121_two_processes_one_directory(backing_dir):
    """Two processes used to name their first file dense_float_0.tmp alike: the second
    truncated the first's data. And one process's sweep deleted the other's live files."""
    from pycauset import _storage

    a, path_a = _child(backing_dir, 7.0)
    b, path_b = _child(backing_dir, 3.0)
    try:
        assert path_a != path_b
        _storage.cleanup_storage(backing_dir)  # our sweep spares their live files
        assert Path(path_a).exists() and Path(path_b).exists()
        for p, want in ((a, 7.0), (b, 3.0)):
            out, _ = p.communicate("\n", timeout=60)
            assert float(out.strip()) == want
    finally:
        for p in (a, b):
            if p.poll() is None:
                p.kill()
    # A dead process's leftovers are swept.
    stale = backing_dir / "dense_float_999999999_0.tmp"
    stale.write_bytes(b"x")
    _storage.cleanup_storage(backing_dir)
    assert not stale.exists()


def test_F122_save_of_a_written_disk_object(backing_dir, tmp_path):
    """msync used an unaligned start (EINVAL, ignored), so flushes did nothing. It is aligned
    and checked now (a failure raises); save flushes the source first."""
    disk = pc.matrix(np.arange(400.0).reshape(20, 20), storage="disk")
    disk[3, 4] = -1.0
    pc.save(disk, tmp_path / "d.pycauset")
    back = pc.load(tmp_path / "d.pycauset")
    assert float(np.asarray(back)[3, 4]) == -1.0


def test_F120_no_object_moves_to_disk_after_creation(backing_dir):
    """The governor used to spill live RAM objects to files to make room, while kernels held
    pointers into them. An object's placement is now fixed when it is made."""
    previous = pc.get_memory_threshold()
    try:
        small = [pc.matrix(np.full((16, 16), float(i)), storage="ram") for i in range(8)]
        pc.set_memory_threshold(1)  # new objects go to files; existing ones stay put
        big = pc.zeros((64, 64))
        assert big.get_backing_file() != ":memory:"
        assert all(m.get_backing_file() == ":memory:" for m in small)
        assert [float(np.asarray(m)[0, 0]) for m in small] == [float(i) for i in range(8)]
    finally:
        pc.set_memory_threshold(previous)


def test_F122_prefetch_and_discard_keep_values(backing_dir, tmp_path):
    """The madvise hints were EINVAL no-ops (unaligned start). Now they run, so discard must
    only drop pages that read back: shared file pages, never a loaded snapshot's writes."""
    m = pc.matrix(np.arange(512 * 512, dtype=np.float64).reshape(512, 512), storage="disk")
    acc = m.get_accelerator()
    acc.prefetch(0, 512 * 512 * 8)
    acc.discard(0, 512 * 512 * 8)
    np.testing.assert_array_equal(np.asarray(m), np.arange(512 * 512.0).reshape(512, 512))
    pc.save(m, tmp_path / "s.pycauset")
    s = pc.load(tmp_path / "s.pycauset")
    s[0, 0] = -5.0
    s.get_accelerator().discard(0, 512 * 512 * 8)
    assert float(np.asarray(s)[0, 0]) == -5.0


# --- area 3.1 review: the storage owns its file (F-123) ------------------------------------

@pytest.fixture
def file_backed(backing_dir):
    previous = pc.get_memory_threshold()
    pc.set_memory_threshold(1)  # every new object is file-backed
    yield backing_dir
    pc.set_memory_threshold(previous)


def _tmp_files(d):
    return sorted(p.name for p in Path(d).glob("*.tmp"))


def test_F123_a_temp_file_goes_with_its_last_owner(file_backed):
    import gc

    a = pc.matrix(np.ones((32, 32)))
    b = a + a
    v = b.T  # a view: shares b's storage and file
    assert len(_tmp_files(file_backed)) == 2
    del a, b
    gc.collect()
    assert len(_tmp_files(file_backed)) == 1  # a's is gone; b's lives on in v
    np.testing.assert_array_equal(np.asarray(v), np.full((32, 32), 2.0))
    del v
    gc.collect()
    assert _tmp_files(file_backed) == []


def test_F123_named_files_and_snapshots_stay(file_backed, tmp_path):
    import gc

    pc.save(pc.matrix(np.eye(4)), tmp_path / "s.pycauset")
    s = pc.load(tmp_path / "s.pycauset")
    s[0, 0] = 9.0  # a private copy; the snapshot file is untouched
    del s
    gc.collect()
    assert (tmp_path / "s.pycauset").exists()
    np.testing.assert_array_equal(np.asarray(pc.load(tmp_path / "s.pycauset")), np.eye(4))


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
def test_F123_file_backed_objects_hold_no_descriptor(file_backed):
    """One open fd per file-backed object: about 1,020 objects hit the default limit."""
    import os

    before = len(os.listdir("/proc/self/fd"))
    keep = [pc.zeros((4, 4)) for _ in range(200)]
    assert all(m.get_backing_file() != ":memory:" for m in keep)
    assert len(os.listdir("/proc/self/fd")) - before < 10


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="needs /proc")
def test_F123_random_causal_matrix_leaks_no_mapping(file_backed):
    """fill_random mapped 64 MB windows and never unmapped them on POSIX."""
    import gc

    for _ in range(3):
        m = pc.TriangularMatrix.random(2000, p=0.5, seed=1)
        del m
    gc.collect()
    maps = Path("/proc/self/maps").read_text()
    assert str(file_backed) not in maps


def test_F123_random_causal_matrix_same_in_ram_and_file(backing_dir):
    previous = pc.get_memory_threshold()
    try:
        for p in (0.5, 0.3):
            ram = np.asarray(pc.TriangularMatrix.random(300, p=p, seed=7))
            pc.set_memory_threshold(1)
            disk_m = pc.TriangularMatrix.random(300, p=p, seed=7)
            assert disk_m.get_backing_file() != ":memory:"
            np.testing.assert_array_equal(np.asarray(disk_m), ram)
            pc.set_memory_threshold(previous)
            assert abs(ram[np.triu_indices(300, 1)].mean() - p) < 0.02
    finally:
        pc.set_memory_threshold(previous)


def test_F123_copy_of_a_small_loaded_matrix_goes_to_ram(backing_dir, tmp_path):
    """Writing to a loaded snapshot that a view shares copied it into a file (never deleted),
    however small. Copies are placed like new objects now."""
    pc.save(pc.matrix(np.eye(4)), tmp_path / "a.pycauset")
    m = pc.load(tmp_path / "a.pycauset")
    v = m.T
    m[0, 0] = 5.0
    assert m.get_backing_file() == ":memory:"
    assert float(np.asarray(v)[0, 0]) == 1.0 and float(np.asarray(m)[0, 0]) == 5.0
