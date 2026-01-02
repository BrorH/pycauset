import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pycauset


class TestNumpyImportMaxInRamBytes(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_threshold = pycauset.get_memory_threshold()
        self._orig_root = pycauset._storage_root()
        self._tmpdir = tempfile.TemporaryDirectory()
        pycauset.set_backing_dir(self._tmpdir.name)

    def tearDown(self) -> None:
        try:
            pycauset.set_backing_dir(self._orig_root)
        except Exception:
            pass
        try:
            pycauset.set_memory_threshold(self._orig_threshold)
        except Exception:
            pass
        self._tmpdir.cleanup()

    def test_matrix_numpy_import_forces_disk_backing_when_limit_exceeded(self) -> None:
        arr = np.arange(16, dtype=np.float64).reshape(4, 4)

        # Sanity: default should be in-memory for tiny inputs.
        m0 = pycauset.matrix(arr)
        try:
            self.assertEqual(m0.get_backing_file(), ":memory:")
        finally:
            m0.close()

        m1 = pycauset.matrix(arr, max_in_ram_bytes=1)
        try:
            bf = m1.get_backing_file()
            self.assertNotEqual(bf, ":memory:")
            p = Path(bf).resolve()
            self.assertTrue(p.exists())
            self.assertEqual(p.parent, Path(self._tmpdir.name).resolve())
        finally:
            m1.close()

    def test_vector_numpy_import_forces_disk_backing_when_limit_exceeded(self) -> None:
        arr = np.arange(32, dtype=np.float64)

        v0 = pycauset.vector(arr)
        try:
            self.assertEqual(v0.get_backing_file(), ":memory:")
        finally:
            v0.close()

        v1 = pycauset.vector(arr, max_in_ram_bytes=1)
        try:
            bf = v1.get_backing_file()
            self.assertNotEqual(bf, ":memory:")
            p = Path(bf).resolve()
            self.assertTrue(p.exists())
            self.assertEqual(p.parent, Path(self._tmpdir.name).resolve())
        finally:
            v1.close()


if __name__ == "__main__":
    unittest.main()
