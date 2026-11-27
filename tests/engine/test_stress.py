"""Stress, fuzzing and leak checks: behaviour that line coverage cannot see.

A threaded test runs the same lines as a single-threaded one, and a fuzz loop the same lines
as one bad file, so coverage-based retirement (TESTS.md §4) scored the old safety suite as
redundant. Its substance lives here, sized for the default tier (seconds).
"""
from __future__ import annotations

import gc
import os
import threading

import numpy as np
import psutil
import pytest

import pycauset as pc
from pycauset._internal import persistence as P


def test_header_fuzzing_never_loads_garbage(tmp_path):
    path = tmp_path / "f.pycauset"
    pc.save(pc.zeros((10, 10), dtype="float64"), str(path))
    rng = np.random.default_rng(0)
    for _ in range(50):
        garbage = rng.integers(0, 256, 64, dtype=np.uint8).tobytes()
        if garbage.startswith(b"PYCAUSET"):
            continue
        with open(path, "r+b") as f:
            f.write(garbage)
        with pytest.raises(ValueError):
            pc.load(str(path))


def test_save_writes_the_payload_to_disk(tmp_path):
    """Read the payload back with plain file I/O: no library cache in between."""
    path = tmp_path / "p.pycauset"
    m = pc.ones((100, 100), dtype="float64")
    m[0, 0], m[99, 99] = 123.456, 789.012
    pc.save(m, str(path))
    active, _, _ = P._read_active_slot_and_typed_metadata(path)
    with open(path, "rb") as f:
        f.seek(active["payload_offset"])
        raw = np.frombuffer(f.read(100 * 100 * 8), dtype=np.float64).reshape(100, 100)
    assert raw[0, 0] == 123.456 and raw[99, 99] == 789.012 and raw[1, 1] == 1.0


def test_threaded_create_save_load_delete(tmp_path):
    """Concurrent construction / save / load / delete. The old suite ran this on Windows only
    ("Linux/macOS concurrency unverified"); it runs everywhere now."""
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            for i in range(10):
                path = tmp_path / f"t{tid}_{i}.pycauset"
                m = pc.matrix(np.full((50, 50), float(tid * 100 + i)))
                pc.save(m, str(path))
                back = pc.load(str(path))
                if np.asarray(back)[0, 0] != tid * 100 + i:
                    raise AssertionError(f"thread {tid}: wrong value after load")
                del back, m
                os.remove(path)
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]


def test_repeated_large_allocations_do_not_accumulate():
    """Five 200 MB alloc/free cycles must not keep the memory (the RAM release path)."""
    proc = psutil.Process(os.getpid())
    n = int((200 * 1024**2 / 8) ** 0.5)
    gc.collect()
    start = proc.memory_info().rss
    for _ in range(5):
        m = pc.ones((n, n), dtype="float64")  # ones: every page is written
        del m
        gc.collect()
    assert proc.memory_info().rss - start < 100 * 1024**2
