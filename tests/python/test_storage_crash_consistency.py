import os
import shutil
import unittest
from pathlib import Path

import pycauset

from pycauset._internal import persistence as _persistence


def _select_active_slot_name(path: Path) -> str:
    active, active_name, _ = _persistence._read_active_slot_and_typed_metadata(path)
    if active is None:
        raise AssertionError("no active slot")
    return str(active_name)


def _flip_byte(path: Path, offset: int) -> None:
    with path.open("rb+") as f:
        f.seek(offset)
        b = f.read(1)
        if len(b) != 1:
            raise AssertionError("failed to read byte")
        f.seek(offset)
        f.write(bytes([b[0] ^ 0xFF]))


def _append_corrupt_metadata_and_point_header(path: Path) -> None:
    file_size = path.stat().st_size

    with path.open("rb+") as f:
        f.seek(_persistence._SLOT_A_OFFSET)
        slot_a = _persistence._unpack_slot(f.read(_persistence._SLOT_SIZE))
        f.seek(_persistence._SLOT_B_OFFSET)
        slot_b = _persistence._unpack_slot(f.read(_persistence._SLOT_SIZE))

        def _slot_valid(s: dict[str, int]) -> bool:
            if not s.get("crc_ok"):
                return False
            if s["payload_offset"] % 4096 != 0:
                return False
            if s["metadata_offset"] % 16 != 0:
                return False
            if s["payload_offset"] + s["payload_length"] > file_size:
                return False
            if s["metadata_offset"] + s["metadata_length"] > file_size:
                return False
            return True

        candidates = [s for s in (slot_a, slot_b) if _slot_valid(s)]
        if not candidates:
            raise AssertionError("no valid slot to extend")

        active = max(candidates, key=lambda s: int(s["generation"]))
        active_is_a = active is slot_a
        active_name = "A" if active_is_a else "B"
        inactive_offset = _persistence._SLOT_B_OFFSET if active_is_a else _persistence._SLOT_A_OFFSET

        # Read existing metadata payload to reuse it
        f.seek(int(active["metadata_offset"]))
        block = f.read(int(active["metadata_length"]))
        if len(block) < 32 or block[:4] != _persistence._METADATA_BLOCK_MAGIC:
            raise AssertionError("metadata block magic mismatch")

        payload_len = int(_persistence.struct.unpack("<Q", block[16:24])[0])
        payload = block[32 : 32 + payload_len]

        # Create a new block with an intentionally bad CRC
        payload_crc = 0x12345678  # wrong on purpose
        block_header = (
            _persistence._METADATA_BLOCK_MAGIC
            + _persistence.struct.pack(
                "<III",
                _persistence._METADATA_BLOCK_VERSION,
                _persistence._METADATA_ENCODING_VERSION,
                0,
            )
            + _persistence.struct.pack("<Q", len(payload))
            + _persistence.struct.pack("<I", payload_crc)
            + _persistence.struct.pack("<I", 0)
        )

        f.seek(0, os.SEEK_END)
        cur = f.tell()
        new_meta_offset = _persistence._align_up(cur, 16)
        if new_meta_offset != cur:
            f.write(b"\x00" * (new_meta_offset - cur))

        f.write(block_header)
        f.write(payload)
        new_meta_len = 32 + len(payload)

        new_slot = _persistence._pack_slot(
            generation=int(active["generation"]) + 1,
            payload_offset=int(active["payload_offset"]),
            payload_length=int(active["payload_length"]),
            metadata_offset=int(new_meta_offset),
            metadata_length=int(new_meta_len),
            hot_offset=int(active.get("hot_offset", 0)),
            hot_length=int(active.get("hot_length", 0)),
        )

        f.seek(inactive_offset)
        f.write(new_slot)


class TestStorageCrashConsistency(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_storage_crash_tmp")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_slot_fallback_to_b_when_a_corrupted(self):
        m = pycauset.FloatMatrix(4)
        path = self.test_dir / "m.pycauset"
        try:
            pycauset.save(m, path)
            self.assertEqual(_select_active_slot_name(path), "A")

            # Corrupt slot A so slot B must be selected.
            _flip_byte(path, _persistence._SLOT_A_OFFSET)

            self.assertEqual(_select_active_slot_name(path), "B")

            loaded = pycauset.load(path)
            loaded.close()
        finally:
            m.close()

    def test_appending_garbage_does_not_change_load(self):
        m = pycauset.IntegerMatrix(8)
        path = self.test_dir / "i.pycauset"
        try:
            pycauset.save(m, path)
            with path.open("ab") as f:
                f.write(b"GARBAGE" * 128)

            loaded = pycauset.load(path)
            loaded.close()
        finally:
            m.close()

    def test_corrupt_new_metadata_block_causes_failure(self):
        m = pycauset.FloatMatrix(3)
        path = self.test_dir / "bad_meta.pycauset"
        try:
            pycauset.save(m, path)

            # Simulate crash sequence where a newer slot points at an invalid metadata block.
            _append_corrupt_metadata_and_point_header(path)

            with self.assertRaises(ValueError) as ctx:
                pycauset.load(path)
            self.assertIn("metadata payload checksum failed", str(ctx.exception))
        finally:
            m.close()

    def test_both_slots_invalid_fails(self):
        m = pycauset.FloatMatrix(2)
        path = self.test_dir / "both_bad.pycauset"
        try:
            pycauset.save(m, path)
            _flip_byte(path, _persistence._SLOT_A_OFFSET)
            _flip_byte(path, _persistence._SLOT_B_OFFSET)

            with self.assertRaises(ValueError) as ctx:
                pycauset.load(path)
            self.assertIn("no valid header slot", str(ctx.exception))
        finally:
            m.close()


if __name__ == "__main__":
    unittest.main()
