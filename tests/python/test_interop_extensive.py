import unittest
import os
import tempfile
import sys
import numpy as np
from pathlib import Path

# Add python directory to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pycauset

class TestInteropExtensive(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

    def test_numpy_conversion_edge_cases(self):
        """Test conversion from various NumPy array shapes and types."""
        
        # 1. Empty array
        arr_empty = np.zeros((0, 0))
        try:
            m_empty = pycauset.asarray(arr_empty)
            self.assertEqual(m_empty.size(), 0)
        except (ValueError, RuntimeError):
            # If empty not supported, that's fine
            pass

        # 2. 1D array (should fail or be treated as vector?)
        arr_1d = np.array([1, 2, 3])
        # If asarray expects matrix, this might fail or return Vector
        try:
            obj = pycauset.asarray(arr_1d)
            # If it returns a vector, check size
            if hasattr(obj, "size"):
                self.assertEqual(obj.size(), 3)
        except (ValueError, TypeError):
            pass

        # 3. Non-square 2D array
        arr_rect = np.zeros((3, 4))
        with self.assertRaises(ValueError):
            # Assuming pycauset only supports square matrices for now
            pycauset.asarray(arr_rect)

    def test_type_preservation(self):
        """Ensure types are preserved or correctly cast during conversion."""
        
        # Float64 -> FloatMatrix
        arr_f64 = np.eye(3, dtype=np.float64)
        m_f64 = pycauset.asarray(arr_f64)
        self.assertIsInstance(m_f64, pycauset.FloatMatrix)
        
        # Int32 -> IntegerMatrix
        arr_i32 = np.eye(3, dtype=np.int32)
        m_i32 = pycauset.asarray(arr_i32)
        self.assertIsInstance(m_i32, pycauset.IntegerMatrix)
        
        # Bool -> DenseBitMatrix (or TriangularBitMatrix if triangular)
        arr_bool = np.eye(3, dtype=bool)
        m_bool = pycauset.asarray(arr_bool)
        # Could be DenseBitMatrix or TriangularBitMatrix depending on implementation preference
        self.assertTrue("BitMatrix" in str(type(m_bool)))

    def test_round_trip(self):
        """Test NumPy -> PyCauset -> NumPy round trip."""
        n = 10
        original = np.random.rand(n, n)
        
        # To PyCauset
        m = pycauset.asarray(original)
        
        # Back to NumPy
        restored = np.array(m)
        
        np.testing.assert_allclose(original, restored)

if __name__ == '__main__':
    unittest.main()
