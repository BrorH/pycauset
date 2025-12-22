import unittest
import os
import shutil
from pathlib import Path
import pycauset
from pycauset import CausalSet, TriangularBitMatrix, IntegerMatrix, FloatMatrix

from pycauset._internal import persistence as _persistence


def _assert_is_container(path: Path) -> None:
    with path.open("rb") as f:
        magic = f.read(8)
    if magic != b"PYCAUSET":
        raise AssertionError(f"expected PYCAUSET magic, got {magic!r}")


def _corrupt_container_cols(path: Path, *, cols: int) -> None:
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
            raise AssertionError("no valid slot to corrupt")

        active = max(candidates, key=lambda s: s["generation"])
        active_is_a = active is slot_a
        inactive_offset = _persistence._SLOT_B_OFFSET if active_is_a else _persistence._SLOT_A_OFFSET

        # Read existing metadata block
        f.seek(active["metadata_offset"])
        block = f.read(active["metadata_length"])
        if block[:4] != _persistence._METADATA_BLOCK_MAGIC:
            raise AssertionError("metadata block magic mismatch")

        payload_len = int(_persistence.struct.unpack("<Q", block[16:24])[0])
        payload = block[32 : 32 + payload_len]
        typed_meta = _persistence._decode_metadata_top_map(payload)
        typed_meta["cols"] = int(cols)

        payload2 = _persistence._encode_metadata_top_map(typed_meta)
        payload_crc = _persistence._crc32(payload2)
        block_header = (
            _persistence._METADATA_BLOCK_MAGIC
            + _persistence.struct.pack(
                "<III",
                _persistence._METADATA_BLOCK_VERSION,
                _persistence._METADATA_ENCODING_VERSION,
                0,
            )
            + _persistence.struct.pack("<Q", len(payload2))
            + _persistence.struct.pack("<I", payload_crc)
            + _persistence.struct.pack("<I", 0)
        )

        f.seek(0, os.SEEK_END)
        cur = f.tell()
        new_meta_offset = _persistence._align_up(cur, 16)
        if new_meta_offset != cur:
            f.write(b"\x00" * (new_meta_offset - cur))

        f.write(block_header)
        f.write(payload2)
        new_meta_len = 32 + len(payload2)

        new_slot = _persistence._pack_slot(
            generation=int(active["generation"]) + 1,
            payload_offset=int(active["payload_offset"]),
            payload_length=int(active["payload_length"]),
            metadata_offset=int(new_meta_offset),
            metadata_length=int(new_meta_len),
            hot_offset=0,
            hot_length=0,
        )

        f.seek(inactive_offset)
        f.write(new_slot)

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_storage_tmp")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_matrix_save_load(self):
        # Create a matrix
        n = 100
        matrix = TriangularBitMatrix.random(n, 0.5, seed=42)
        path = self.test_dir / "matrix.pycauset"
        
        try:
            # Save
            pycauset.save(matrix, path)
            
            # Verify file exists
            self.assertTrue(path.exists())
            
            _assert_is_container(path)
            meta, _ = _persistence._read_new_container_metadata_and_offset(path)
            self.assertEqual(meta["rows"], n)
            self.assertEqual(meta["matrix_type"], "CAUSAL")
                    
            # Load
            loaded_matrix = pycauset.load(path)
            try:
                self.assertIsInstance(loaded_matrix, TriangularBitMatrix)
                self.assertEqual(loaded_matrix.rows(), n)
                self.assertEqual(loaded_matrix.cols(), n)
            finally:
                loaded_matrix.close()
        finally:
            matrix.close()
        
    def test_causet_save_load(self):
        # Create CausalSet
        c = CausalSet(n=50, seed=123)
        path = self.test_dir / "causet.pycauset"
        
        try:
            # Save
            c.save(path)

            _assert_is_container(path)
            meta, _ = _persistence._read_new_container_metadata_and_offset(path)
            self.assertEqual(meta.get("object_type"), "CausalSet")
            self.assertEqual(meta.get("n"), 50)
                    
            # Load
            loaded_c = CausalSet.load(path)
            try:
                self.assertIsInstance(loaded_c, CausalSet)
                self.assertEqual(loaded_c.n, 50)
                self.assertEqual(loaded_c.C.rows(), 50)
                self.assertEqual(loaded_c.C.cols(), 50)
            finally:
                loaded_c.C.close()
        finally:
            c.C.close()

    def test_integer_matrix_save_load(self):
        n = 10
        matrix = IntegerMatrix(n) # Zero initialized
        path = self.test_dir / "int_matrix.pycauset"
        
        try:
            pycauset.save(matrix, path)
            
            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, IntegerMatrix)
                self.assertEqual(loaded.rows(), n)
                self.assertEqual(loaded.cols(), n)
            finally:
                loaded.close()
        finally:
            matrix.close()

    def test_dense_bit_matrix_rectangular_save_load(self):
        if getattr(pycauset, "DenseBitMatrix", None) is None:
            self.skipTest("DenseBitMatrix is not available")

        m = pycauset.DenseBitMatrix(3, 5)
        m.set(1, 4, True)
        m.set(2, 0, True)
        path = self.test_dir / "bit_matrix_rect.pycauset"

        try:
            pycauset.save(m, path)

            meta, _ = _persistence._read_new_container_metadata_and_offset(path)
            self.assertEqual(meta["rows"], 3)
            self.assertEqual(meta["cols"], 5)
            self.assertEqual(meta.get("data_type"), "BIT")

            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, pycauset.DenseBitMatrix)
                self.assertEqual(loaded.rows(), 3)
                self.assertEqual(loaded.cols(), 5)
                self.assertTrue(loaded.get(1, 4))
                self.assertTrue(loaded.get(2, 0))
                self.assertFalse(loaded.get(0, 0))
            finally:
                loaded.close()
        finally:
            m.close()

    def test_square_only_metadata_mismatch_rejected(self):
        n = 16
        matrix = TriangularBitMatrix.random(n, 0.25, seed=7)
        path = self.test_dir / "triangular_ok.pycauset"
        bad_path = self.test_dir / "triangular_bad_cols.pycauset"

        try:
            pycauset.save(matrix, path)

            shutil.copy2(path, bad_path)
            _corrupt_container_cols(bad_path, cols=n + 1)

            with self.assertRaises(ValueError):
                pycauset.load(bad_path)
        finally:
            matrix.close()

    def test_int16_matrix_save_load(self):
        if getattr(pycauset, "Int16Matrix", None) is None:
            self.skipTest("Int16Matrix is not available")

        n = 10
        matrix = pycauset.Int16Matrix(n)
        matrix[0, 0] = 7
        matrix[1, 0] = -3
        path = self.test_dir / "int16_matrix.pycauset"

        try:
            pycauset.save(matrix, path)

            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, pycauset.Int16Matrix)
                self.assertEqual(loaded.rows(), n)
                self.assertEqual(loaded.cols(), n)
                self.assertEqual(loaded[0, 0], 7)
                self.assertEqual(loaded[1, 0], -3)
            finally:
                loaded.close()
        finally:
            matrix.close()

    def test_int16_vector_save_load(self):
        if getattr(pycauset, "Int16Vector", None) is None:
            self.skipTest("Int16Vector is not available")

        n = 10
        vec = pycauset.Int16Vector(n)
        vec[0] = 4
        vec[1] = -5
        path = self.test_dir / "int16_vector.pycauset"

        try:
            pycauset.save(vec, path)

            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, pycauset.Int16Vector)
                self.assertEqual(len(loaded), n)
                self.assertEqual(loaded[0], 4)
                self.assertEqual(loaded[1], -5)
            finally:
                loaded.close()
        finally:
            vec.close()

if __name__ == "__main__":
    unittest.main()
