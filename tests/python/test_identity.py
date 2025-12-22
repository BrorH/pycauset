import unittest
import os
import tempfile
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Set up storage dir before importing pycauset
_STORAGE_TMP = tempfile.TemporaryDirectory()
os.environ["PYCAUSET_STORAGE_DIR"] = _STORAGE_TMP.name

import pycauset


class TestIdentityFactory(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        _STORAGE_TMP.cleanup()

    def test_identity_from_int(self):
        m = pycauset.identity(4)
        self.assertEqual(m.shape, (4, 4))
        self.assertEqual(m[0, 0], 1.0)
        self.assertEqual(m[1, 1], 1.0)
        self.assertEqual(m[0, 1], 0.0)
        self.assertEqual(m[3, 2], 0.0)

    def test_identitymatrix_from_int(self):
        m = pycauset.IdentityMatrix(4)
        self.assertEqual(m.shape, (4, 4))
        self.assertEqual(m[0, 0], 1.0)
        self.assertEqual(m[0, 1], 0.0)

    def test_identity_from_shape_list_rectangular(self):
        m = pycauset.identity([2, 5])
        self.assertEqual(m.shape, (2, 5))
        self.assertEqual(m[0, 0], 1.0)
        self.assertEqual(m[1, 1], 1.0)
        self.assertEqual(m[0, 3], 0.0)
        self.assertEqual(m[1, 4], 0.0)

    def test_identitymatrix_from_shape_list_rectangular(self):
        m = pycauset.IdentityMatrix([2, 5])
        self.assertEqual(m.shape, (2, 5))
        self.assertEqual(m[0, 0], 1.0)
        self.assertEqual(m[1, 1], 1.0)
        self.assertEqual(m[1, 4], 0.0)

    def test_identity_from_matrix(self):
        a = pycauset.IntegerMatrix(3, 2)
        m = pycauset.identity(a)
        self.assertEqual(m.shape, (3, 2))
        self.assertEqual(m[0, 0], 1.0)
        self.assertEqual(m[1, 1], 1.0)
        self.assertEqual(m[2, 0], 0.0)
        self.assertEqual(m[2, 1], 0.0)

    def test_identitymatrix_from_matrix(self):
        a = pycauset.IntegerMatrix(3, 2)
        m = pycauset.IdentityMatrix(a)
        self.assertEqual(m.shape, (3, 2))
        self.assertEqual(m[0, 0], 1.0)
        self.assertEqual(m[1, 1], 1.0)
        self.assertEqual(m[2, 1], 0.0)

    def test_identity_from_vector(self):
        v = pycauset.IntegerVector(6)
        m = pycauset.identity(v)
        self.assertEqual(m.shape, (6, 6))
        self.assertEqual(m[5, 5], 1.0)
        self.assertEqual(m[0, 5], 0.0)

    def test_identitymatrix_from_vector(self):
        v = pycauset.IntegerVector(6)
        m = pycauset.IdentityMatrix(v)
        self.assertEqual(m.shape, (6, 6))
        self.assertEqual(m[5, 5], 1.0)
        self.assertEqual(m[0, 5], 0.0)

    def test_identity_persistence_roundtrip_rectangular(self):
        m = pycauset.identity([3, 5])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "id_3x5.pycauset"
            pycauset.save(m, path)
            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, pycauset.IdentityMatrix)
                self.assertEqual(loaded.shape, (3, 5))
                self.assertEqual(loaded[0, 0], 1.0)
                self.assertEqual(loaded[2, 2], 1.0)
                self.assertEqual(loaded[2, 4], 0.0)
            finally:
                loaded.close()


if __name__ == "__main__":
    unittest.main()
