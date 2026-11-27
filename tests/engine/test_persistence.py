"""Container robustness and the other persistence paths (TESTS.md layout: test_persistence.py).

Round trips of every structure x dtype x view are the atlas's `save_load` op; this file pins
what the atlas can't express: crash consistency of the A/B header slots (Container Format
doc), causal sets, and the .npy / .npz / convert_file helpers. Moved from the old suite.
"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest

import pycauset as pc
from pycauset._internal import persistence as P
from tests.conftest import to_np


def _active(path) -> str:
    return P._read_active_slot_and_typed_metadata(path)[1]


def _flip_byte(path, offset: int) -> None:
    with open(path, "rb+") as f:
        f.seek(offset)
        b = f.read(1)
        f.seek(offset)
        f.write(bytes([b[0] ^ 0xFF]))


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / "m.pycauset"
    data = np.arange(16.0).reshape(4, 4)
    pc.save(pc.matrix(data), str(path))
    return path, data


def test_corrupt_slot_a_falls_back_to_b(saved):
    path, data = saved
    assert _active(path) == "A"
    _flip_byte(path, P._SLOT_A_OFFSET)
    assert _active(path) == "B"
    np.testing.assert_array_equal(to_np(pc.load(str(path))), data)


def test_both_slots_corrupt_fails(saved):
    path, _ = saved
    _flip_byte(path, P._SLOT_A_OFFSET)
    _flip_byte(path, P._SLOT_B_OFFSET)
    with pytest.raises(ValueError, match="no valid header slot"):
        pc.load(str(path))


def test_trailing_garbage_does_not_change_load(saved):
    path, data = saved
    with open(path, "ab") as f:
        f.write(b"GARBAGE" * 128)
    np.testing.assert_array_equal(to_np(pc.load(str(path))), data)


def test_newer_slot_over_a_corrupt_metadata_block_fails_loudly(saved):
    """The writer writes metadata before the header (Container Format), so a valid newer
    header over a bad metadata block is corruption, not a crash: load raises."""
    path, _ = saved
    active, name, _ = P._read_active_slot_and_typed_metadata(path)
    with open(path, "rb+") as f:
        f.seek(active["metadata_offset"])
        block = f.read(active["metadata_length"])
        payload = block[32:]
        bad = P._METADATA_BLOCK_MAGIC + struct.pack("<III", P._METADATA_BLOCK_VERSION, P._METADATA_ENCODING_VERSION, 0)
        bad += struct.pack("<QII", len(payload), 0x12345678, 0) + payload  # wrong CRC
        f.seek(0, os.SEEK_END)
        offset = P._align_up(f.tell(), 16)
        f.write(b"\0" * (offset - f.tell()) + bad)
        f.seek(P._SLOT_B_OFFSET if name == "A" else P._SLOT_A_OFFSET)
        f.write(P._pack_slot(generation=active["generation"] + 1, payload_offset=active["payload_offset"],
                             payload_length=active["payload_length"], metadata_offset=offset,
                             metadata_length=len(bad), hot_offset=active.get("hot_offset", 0),
                             hot_length=active.get("hot_length", 0)))
    with pytest.raises(ValueError, match="metadata payload checksum failed"):
        pc.load(str(path))


def test_causal_set_round_trip(tmp_path):
    c = pc.CausalSet(30, spacetime=pc.spacetime.MinkowskiDiamond(2), seed=4)
    path = tmp_path / "c.pycauset"
    c.save(str(path))
    back = pc.CausalSet.load(str(path))
    assert back.n == 30
    np.testing.assert_array_equal(np.asarray(back.C), np.asarray(c.C))


@pytest.mark.parametrize("dtype", ["float64", "int16", "bool", "complex128"])
def test_npy_and_npz_round_trip(tmp_path, dtype):
    x = (np.arange(12).reshape(3, 4) % 2).astype(dtype)
    m = pc.matrix(x)
    npy, npz = tmp_path / "x.npy", tmp_path / "x.npz"
    pc.save_npy(m, str(npy))
    pc.save_npz(m, str(npz))
    np.testing.assert_array_equal(np.load(npy), x)
    np.testing.assert_array_equal(to_np(pc.load_npy(str(npy))), x)
    np.testing.assert_array_equal(to_np(pc.load_npz(str(npz))), x)


def test_convert_file_between_formats(tmp_path):
    x = np.arange(9.0).reshape(3, 3)
    src = tmp_path / "x.npy"
    np.save(src, x)
    dst = tmp_path / "x.pycauset"
    pc.convert_file(str(src), str(dst))
    np.testing.assert_array_equal(to_np(pc.load(str(dst))), x)
    back = tmp_path / "back.npz"
    pc.convert_file(str(dst), str(back))
    np.testing.assert_array_equal(np.load(back)["array"], x)
    with pytest.raises(ValueError):
        pc.convert_file(str(src), str(tmp_path / "x.unknown"))


def test_corrupt_metadata_magic_fails(saved):
    path, _ = saved
    active, _, _ = P._read_active_slot_and_typed_metadata(path)
    _flip_byte(path, active["metadata_offset"])  # first byte of the block magic
    with pytest.raises(ValueError, match="metadata block magic mismatch|no valid header slot"):
        pc.load(str(path))


def test_causal_set_on_a_cylinder_round_trip(tmp_path):
    st = pc.spacetime.MinkowskiCylinder(2, height=2.0, circumference=3.0)
    c = pc.CausalSet(25, spacetime=st, seed=2)
    path = tmp_path / "cyl.pycauset"
    c.save(str(path))
    back = pc.CausalSet.load(str(path))
    assert type(back.spacetime).__name__ == "MinkowskiCylinder"
    np.testing.assert_array_equal(np.asarray(back.C), np.asarray(c.C))


def test_convert_npz_to_npy_and_one_dimensional_data(tmp_path):
    src = tmp_path / "v.npz"
    np.savez(src, array=np.arange(5.0))
    dst = tmp_path / "v.npy"
    pc.convert_file(str(src), str(dst))
    np.testing.assert_array_equal(np.load(dst), np.arange(5.0))
    v = pc.load_npy(str(dst))
    assert isinstance(v, pc.VectorBase)


@pytest.mark.parametrize("make", [pc.matrix, pc.vector])
def test_max_in_ram_bytes_imports_large_arrays_to_disk(make, tmp_path):
    x = np.arange(64.0).reshape(8, 8) if make is pc.matrix else np.arange(64.0)
    obj = make(x, max_in_ram_bytes=64)  # 512 bytes > 64: file-backed import
    assert obj.get_backing_file() != ":memory:"
    np.testing.assert_array_equal(pc.to_numpy(obj, allow_huge=True), x)
    small = make(x, max_in_ram_bytes=1 << 20)
    assert small.get_backing_file() == ":memory:"
