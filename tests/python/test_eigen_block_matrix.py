"""
Test Phase 6: Block-Matrix Support for Eigen Operations

Verifies that eigen operations properly handle block matrices:
- Reject block matrices with clear error messages (not supported)
- Verify OpContract declarations match actual behavior
"""

import unittest
import pycauset
import numpy as np


class TestEigenBlockMatrixSupport(unittest.TestCase):
    def setUp(self):
        """Create a simple 2x2 block matrix for testing"""
        # Create 4 blocks of 2x2 matrices
        self.block_00 = pycauset.matrix(np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64))
        self.block_01 = pycauset.matrix(np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float64))
        self.block_10 = pycauset.matrix(np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float64))
        self.block_11 = pycauset.matrix(np.array([[4.0, 5.0], [5.0, 6.0]], dtype=np.float64))
        
        # Create block matrix
        try:
            from pycauset._internal.blockmatrix import BlockMatrix
            self.block_matrix = BlockMatrix(
                [[self.block_00, self.block_01],
                 [self.block_10, self.block_11]]
            )
            self.has_blockmatrix = True
        except (ImportError, AttributeError):
            self.has_blockmatrix = False
            
    def test_eigh_rejects_block_matrix(self):
        """Test that eigh rejects block matrices with clear error"""
        if not self.has_blockmatrix:
            self.skipTest("BlockMatrix not available")
            
        # eigh should reject block matrices since LAPACK requires dense contiguous
        with self.assertRaises((NotImplementedError, TypeError, ValueError)) as cm:
            pycauset.eigh(self.block_matrix)
            
        # Error message should mention block matrix or unsupported
        error_msg = str(cm.exception).lower()
        self.assertTrue(
            "block" in error_msg or "unsupported" in error_msg or "dense" in error_msg,
            f"Error message should mention block/unsupported/dense: {error_msg}"
        )
        
    def test_eigvalsh_rejects_block_matrix(self):
        """Test that eigvalsh rejects block matrices"""
        if not self.has_blockmatrix:
            self.skipTest("BlockMatrix not available")
            
        with self.assertRaises((NotImplementedError, TypeError, ValueError)):
            pycauset.eigvalsh(self.block_matrix)
            
    def test_eig_rejects_block_matrix(self):
        """Test that eig (general) rejects block matrices"""
        if not self.has_blockmatrix:
            self.skipTest("BlockMatrix not available")
            
        with self.assertRaises((NotImplementedError, TypeError, ValueError)):
            pycauset.eig(self.block_matrix)
            
    def test_eigvals_rejects_block_matrix(self):
        """Test that eigvals rejects block matrices"""
        if not self.has_blockmatrix:
            self.skipTest("BlockMatrix not available")
            
        with self.assertRaises((NotImplementedError, TypeError, ValueError)):
            pycauset.eigvals(self.block_matrix)
            
    def test_eigvals_arnoldi_rejects_block_matrix(self):
        """Test that eigvals_arnoldi rejects block matrices (currently)"""
        if not self.has_blockmatrix:
            self.skipTest("BlockMatrix not available")
            
        # Arnoldi currently doesn't support block matrices at the top level
        # (though the underlying A could be disk-backed dense)
        with self.assertRaises((NotImplementedError, TypeError, ValueError)):
            pycauset.eigvals_arnoldi(self.block_matrix, k=2, m=4, tol=1e-6)
            
    def test_dense_matrix_works(self):
        """Sanity check: dense matrices should work"""
        # Create a simple 4x4 dense symmetric matrix
        A_np = np.array([[4, 1, 0, 0],
                        [1, 3, 1, 0],
                        [0, 1, 2, 1],
                        [0, 0, 1, 1]], dtype=np.float64)
        A = pycauset.matrix(A_np)
        
        # eigh should work on dense
        w, v = pycauset.eigh(A)
        self.assertEqual(w.size(), 4)
        self.assertEqual(v.rows(), 4)
        self.assertEqual(v.cols(), 4)
        
        # eigvalsh should work on dense
        w2 = pycauset.eigvalsh(A)
        self.assertEqual(w2.size(), 4)


if __name__ == "__main__":
    unittest.main()
