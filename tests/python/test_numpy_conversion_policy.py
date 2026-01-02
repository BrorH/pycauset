import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pycauset


class TestNumpyConversionPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_threshold = pycauset.get_memory_threshold()
        self._orig_export = pycauset._EXPORT_MAX_BYTES
        self._orig_root = pycauset._storage_root()
        self._tmpdir = tempfile.TemporaryDirectory()
        pycauset.set_backing_dir(self._tmpdir.name)
        pycauset.keep_temp_files = False
        pycauset.set_export_max_bytes(None)

    def tearDown(self) -> None:
        try:
            pycauset.set_memory_threshold(self._orig_threshold)
        except Exception:
            pass
        try:
            pycauset.set_export_max_bytes(self._orig_export)
        except Exception:
            pass
        try:
            pycauset.set_backing_dir(self._orig_root)
        except Exception:
            pass
        self._tmpdir.cleanup()

    def test_file_backed_export_blocked_without_allow_huge(self) -> None:
        pycauset.set_memory_threshold(1)
        m = pycauset.TriangularBitMatrix(64)
        try:
            backing = m.get_backing_file()
            self.assertNotEqual(backing, ":memory:")
            with self.assertRaises(RuntimeError):
                np.array(m)
        finally:
            m.close()

    def test_file_backed_export_allowed_with_flag(self) -> None:
        pycauset.set_memory_threshold(1)
        m = pycauset.TriangularBitMatrix(64)
        try:
            arr = pycauset.to_numpy(m, allow_huge=True)
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, (64, 64))
        finally:
            m.close()

    def test_snapshot_export_allowed_without_huge_flag(self) -> None:
        pycauset.set_memory_threshold(1)
        m = pycauset.FloatMatrix(128)
        m.set(0, 0, 1.0)
        snap = Path(self._tmpdir.name) / "snapshot.pycauset"
        try:
            pycauset.save(m, snap)
            loaded = pycauset.load(snap)
            try:
                arr = np.array(loaded)
                self.assertEqual(arr.shape, (128, 128))
                self.assertEqual(arr[0, 0], 1.0)
            finally:
                loaded.close()
        finally:
            m.close()
            if snap.exists():
                snap.unlink()

    def test_export_ceiling_blocks_in_memory_exports(self) -> None:
        m = pycauset.FloatMatrix(64)
        m.set(0, 0, 1.0)
        try:
            pycauset.set_export_max_bytes(1)
            with self.assertRaises(RuntimeError):
                np.array(m)
            with self.assertRaises(RuntimeError):
                pycauset.to_numpy(m)
        finally:
            m.close()
            pycauset.set_export_max_bytes(None)

    def test_allow_huge_overrides_export_ceiling(self) -> None:
        m = pycauset.FloatMatrix(64)
        m.set(0, 0, 1.0)
        try:
            pycauset.set_export_max_bytes(1)
            arr = pycauset.to_numpy(m, allow_huge=True)
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, (64, 64))
            self.assertEqual(arr[0, 0], 1.0)
        finally:
            m.close()
            pycauset.set_export_max_bytes(None)


if __name__ == "__main__":
    unittest.main()
