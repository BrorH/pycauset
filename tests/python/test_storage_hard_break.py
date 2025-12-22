import os
import shutil
import unittest
from pathlib import Path

import pycauset

from pycauset._internal import persistence as _persistence


def _flip_byte(path: Path, offset: int) -> None:
    with path.open("rb+") as f:
        f.seek(offset)
        b = f.read(1)
        if len(b) != 1:
            raise AssertionError("failed to read byte")
        f.seek(offset)
        f.write(bytes([b[0] ^ 0xFF]))


class TestStorageHardBreak(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_storage_hard_break_tmp")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_magic_mismatch_is_not_supported(self):
        path = self.test_dir / "not_a_container.pycauset"
        path.write_bytes(b"NOTPYCAUSET" + b"\x00" * 64)

        with self.assertRaises(ValueError) as ctx:
            pycauset.load(path)
        self.assertIn("unsupported file format", str(ctx.exception))

    def test_invalid_header_slot_is_distinct_error(self):
        m = pycauset.FloatMatrix(4)
        path = self.test_dir / "bad_header.pycauset"
        try:
            pycauset.save(m, path)

            # Corrupt both slots so header selection fails deterministically.
            _flip_byte(path, _persistence._SLOT_A_OFFSET)
            _flip_byte(path, _persistence._SLOT_B_OFFSET)

            with self.assertRaises(ValueError) as ctx:
                pycauset.load(path)
            self.assertIn("no valid header slot", str(ctx.exception))
        finally:
            m.close()

    def test_invalid_metadata_block_is_distinct_error(self):
        m = pycauset.IntegerMatrix(4)
        path = self.test_dir / "bad_metadata.pycauset"
        try:
            pycauset.save(m, path)

            active, _, _ = _persistence._read_active_slot_and_typed_metadata(path)
            meta_offset = int(active["metadata_offset"])

            # Corrupt the metadata block magic.
            _flip_byte(path, meta_offset)

            with self.assertRaises(ValueError) as ctx:
                pycauset.load(path)
            self.assertIn("metadata block magic mismatch", str(ctx.exception))
        finally:
            m.close()


if __name__ == "__main__":
    unittest.main()
