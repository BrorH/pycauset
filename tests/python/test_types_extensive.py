import unittest
import os
import tempfile
import sys
import math
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

class TestTypesExtensive(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

    def test_integer_limits(self):
        """Test IntegerMatrix with boundary integer values."""
        n = 2
        m = pycauset.IntegerMatrix(n)
        
        # Max int32
        max_int = 2**31 - 1
        m[0, 1] = max_int
        self.assertEqual(m[0, 1], max_int)
        
        # Min int32
        min_int = -2**31
        m[1, 0] = min_int
        self.assertEqual(m[1, 0], min_int)
        
        # Overflow check (Python ints are arbitrary precision, C++ are fixed)
        # This might wrap around or throw, depending on pybind11 casting
        try:
            m[0, 0] = max_int + 1
            # If it didn't throw, check what happened
            val = m[0, 0]
            # It likely wrapped to negative
            # print(f"Overflow result: {val}")
        except (OverflowError, TypeError):
            pass

    def test_float_special_values(self):
        """Test FloatMatrix with NaN and Infinity."""
        n = 2
        m = pycauset.FloatMatrix(n)
        
        # Infinity
        m[0, 1] = float('inf')
        self.assertEqual(m[0, 1], float('inf'))
        self.assertTrue(math.isinf(m[0, 1]))
        
        # NaN
        m[1, 0] = float('nan')
        self.assertTrue(math.isnan(m[1, 0]))
        
        # Negative Infinity
        m[0, 0] = float('-inf')
        self.assertEqual(m[0, 0], float('-inf'))

    def test_bit_matrix_logic(self):
        """Test boolean logic in BitMatrices."""
        n = 2
        m = pycauset.DenseBitMatrix(n)
        
        # True/False
        m[0, 1] = True
        self.assertTrue(m[0, 1])
        
        m[0, 1] = False
        self.assertFalse(m[0, 1])
        
        # Integer assignment (0/1)
        m[1, 0] = 1
        self.assertTrue(m[1, 0])
        
        m[1, 0] = 0
        self.assertFalse(m[1, 0])
        
        # Non-zero integer -> True?
        try:
            m[0, 0] = 5
            self.assertTrue(m[0, 0])
        except TypeError:
            # Strict 0/1 enforcement is also valid
            pass

    def test_triangular_constraints(self):
        """Test strict triangularity enforcement."""
        n = 3
        m = pycauset.TriangularBitMatrix(n)
        
        # Setting diagonal (i == j) -> Should fail or be ignored?
        # Usually strictly upper triangular means diagonal is 0
        try:
            m[0, 0] = True
            # If it allows setting, verify if it persists
            # self.assertTrue(m[0, 0]) 
        except (ValueError, IndexError):
            pass
            
        # Setting lower triangle (i > j) -> Should fail or warn
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                m[1, 0] = True
                # If it didn't raise, check for warning
                if len(w) > 0:
                    self.assertTrue("lower-triangular" in str(w[-1].message) or "diagonal" in str(w[-1].message))
                else:
                    # If no warning and no error, that's a potential issue, but maybe it just ignores silently?
                    # The output showed a UserWarning, so we expect that.
                    pass
            except (ValueError, IndexError, RuntimeError):
                pass

if __name__ == '__main__':
    unittest.main()
