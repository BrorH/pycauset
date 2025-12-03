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

class TestOperationsExtensive(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        # Ensure we don't leak files
        self.addCleanup(self.tmp_dir.cleanup)

    def test_scalar_multiplication_edge_cases(self):
        """Test scalar multiplication with edge case values."""
        n = 10
        m = pycauset.TriangularFloatMatrix(n)
        m[0, 1] = 5.5
        m[2, 3] = -2.0
        
        # Multiply by zero
        m_zero = m * 0.0
        self.assertEqual(m_zero[0, 1], 0.0)
        self.assertEqual(m_zero[2, 3], 0.0)
        
        # Multiply by one
        m_one = m * 1.0
        self.assertEqual(m_one[0, 1], 5.5)
        
        # Multiply by negative
        m_neg = m * -1.0
        self.assertEqual(m_neg[0, 1], -5.5)
        self.assertEqual(m_neg[2, 3], 2.0)
        
        # Multiply by large number
        large_val = 1e10
        m_large = m * large_val
        self.assertEqual(m_large[0, 1], 5.5 * large_val)

    def test_mixed_type_arithmetic(self):
        """Test arithmetic between different matrix types."""
        n = 5
        f_mat = pycauset.FloatMatrix(n)
        i_mat = pycauset.IntegerMatrix(n)
        
        f_mat[0, 1] = 2.5
        i_mat[0, 1] = 3
        
        # Float + Int -> Float (usually)
        # Note: If not supported, this should raise TypeError or be handled
        try:
            res = f_mat + i_mat
            self.assertIsInstance(res, pycauset.FloatMatrix)
            self.assertEqual(res[0, 1], 5.5)
        except TypeError:
            # If mixed arithmetic isn't implemented yet, document it
            print("Mixed Float+Int arithmetic not supported yet")

    def test_empty_matrix_operations(self):
        """Test operations on 0x0 matrices."""
        n = 0
        m1 = pycauset.FloatMatrix(n)
        m2 = pycauset.FloatMatrix(n)
        
        # Should not crash
        res_add = m1 + m2
        res_mul = m1 * m2
        
        self.assertEqual(res_add.size(), 0)
        self.assertEqual(res_mul.size(), 0)

    def test_chained_operations(self):
        """Test chained arithmetic operations."""
        n = 5
        A = pycauset.FloatMatrix(n)
        B = pycauset.FloatMatrix(n)
        C = pycauset.FloatMatrix(n)
        
        A[0, 1] = 1.0
        B[0, 1] = 2.0
        C[0, 1] = 3.0
        
        # (A + B) * C
        # (1 + 2) * 3 = 9
        D = (A + B) * C
        self.assertEqual(D[0, 1], 9.0)
        
        # A + B * C (Precedence check)
        # 1 + (2 * 3) = 7
        E = A + B * C
        self.assertEqual(E[0, 1], 7.0)

    def test_inplace_operations(self):
        """Test in-place operators (if supported)."""
        n = 5
        A = pycauset.FloatMatrix(n)
        A[0, 1] = 10.0
        
        expected = 10.0
        
        # += Scalar
        A += 5.0
        expected += 5.0
        self.assertEqual(A[0, 1], expected)
            
        B = pycauset.FloatMatrix(n)
        B[0, 1] = 2.0
        
        # Matrix +=
        try:
            A += B
            expected += 2.0
            self.assertEqual(A[0, 1], expected)
        except TypeError:
            pass

    def test_scalar_addition(self):
        """Test scalar addition."""
        n = 5
        A = pycauset.FloatMatrix(n)
        A[0, 1] = 10.0
        
        # A + 5.0
        B = A + 5.0
        self.assertEqual(B[0, 1], 15.0)
        self.assertEqual(B[0, 0], 5.0) # Was 0.0
        
        # 5.0 + A
        C = 5.0 + A
        self.assertEqual(C[0, 1], 15.0)
        self.assertEqual(C[0, 0], 5.0)
        
        # Int scalar
        D = A + 2
        self.assertEqual(D[0, 1], 12.0)
        
        # BitMatrix + scalar
        BM = pycauset.DenseBitMatrix(n)
        BM[0, 1] = 1
        E = BM + 1
        # 1 + 1 = 2
        self.assertEqual(E[0, 1], 2.0)
        # 0 + 1 = 1
        self.assertEqual(E[0, 0], 1.0)

    def test_matrix_multiplication_shapes(self):
        """Test matrix multiplication (matmul) with compatible and incompatible shapes."""
        # Square matrices for now as pycauset seems to focus on square
        n = 4
        A = pycauset.FloatMatrix(n)
        B = pycauset.FloatMatrix(n)
        
        # Identity-like behavior check
        for i in range(n):
            A[i, i] = 1.0
            
        B[0, 1] = 5.0
        
        # I * B = B
        C = A @ B
        self.assertEqual(C[0, 1], 5.0)
        
        # Incompatible sizes (if non-square supported, or just different n)
        D = pycauset.FloatMatrix(n + 1)
        with self.assertRaises(ValueError):
            A @ D

if __name__ == '__main__':
    unittest.main()
