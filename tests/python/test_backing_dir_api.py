import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pycauset


class TestBackingDirAPI(unittest.TestCase):
    def test_set_backing_dir_affects_new_disk_backed_allocations(self):
        # Force disk-backed allocation even for small matrices.
        orig_threshold = pycauset.get_memory_threshold()
        orig_root = None
        if hasattr(pycauset, "_runtime"):
            try:
                orig_root = pycauset._runtime.storage_root()
            except Exception:
                orig_root = None

        pycauset.set_memory_threshold(1)
        try:
            with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
                root1 = Path(d1).resolve()
                root2 = Path(d2).resolve()

                pycauset.set_backing_dir(root1)
                m1 = pycauset.TriangularBitMatrix(64)
                try:
                    bf1 = m1.get_backing_file()
                    self.assertNotEqual(bf1, ":memory:")
                    p1 = Path(bf1).resolve()
                    self.assertTrue(p1.exists())
                    self.assertEqual(p1.parent, root1)
                finally:
                    m1.close()

                pycauset.set_backing_dir(root2)
                m2 = pycauset.TriangularBitMatrix(64)
                try:
                    bf2 = m2.get_backing_file()
                    self.assertNotEqual(bf2, ":memory:")
                    p2 = Path(bf2).resolve()
                    self.assertTrue(p2.exists())
                    self.assertEqual(p2.parent, root2)
                finally:
                    m2.close()
        finally:
            if orig_threshold is not None:
                pycauset.set_memory_threshold(orig_threshold)
            if orig_root is not None:
                pycauset.set_backing_dir(orig_root)
