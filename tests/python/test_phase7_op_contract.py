"""
Test Phase 7: Op Contract Registration for Additional Operations

Verifies that Phase 7 operations (trace, determinant, norms, factorizations)
are properly registered in the OpRegistry with correct capabilities.
"""

import unittest

try:
    import _pycauset as internal
except ImportError:
    try:
        from pycauset import _pycauset as internal
    except ImportError:
        internal = None


class TestPhase7OpContract(unittest.TestCase):
    def setUp(self):
        if internal is None:
            self.skipTest("_pycauset module not available")
        self.registry = internal.OpRegistry.instance()
        
    def test_trace_contract(self):
        """Test trace operation contract"""
        contract = self.registry.get_contract("trace")
        self.assertIsNotNone(contract, "trace not registered in OpRegistry")
        self.assertEqual(contract.name, "trace")
        self.assertTrue(contract.supports_streaming, "trace should support streaming (diagonal access only)")
        self.assertTrue(contract.supports_block_matrix, "trace can sum block diagonals")
        self.assertFalse(contract.requires_square, "trace works on non-square")
        
    def test_determinant_contract(self):
        """Test determinant operation contract"""
        contract = self.registry.get_contract("determinant")
        self.assertIsNotNone(contract, "determinant not registered in OpRegistry")
        self.assertEqual(contract.name, "determinant")
        self.assertFalse(contract.supports_streaming, "determinant uses LU (full matrix)")
        self.assertTrue(contract.supports_block_matrix, "determinant can use block formula")
        self.assertTrue(contract.requires_square, "determinant requires square matrix")
        
    def test_frobenius_norm_contract(self):
        """Test frobenius_norm operation contract"""
        contract = self.registry.get_contract("frobenius_norm")
        self.assertIsNotNone(contract, "frobenius_norm not registered in OpRegistry")
        self.assertEqual(contract.name, "frobenius_norm")
        self.assertTrue(contract.supports_streaming, "frobenius norm is streaming-safe (sum of squares)")
        self.assertTrue(contract.supports_block_matrix, "frobenius norm can sum block norms")
        self.assertFalse(contract.requires_square, "frobenius norm works on any shape")
        
    def test_qr_contract(self):
        """Test QR decomposition contract"""
        contract = self.registry.get_contract("qr")
        self.assertIsNotNone(contract, "qr not registered in OpRegistry")
        self.assertEqual(contract.name, "qr")
        self.assertFalse(contract.supports_streaming, "QR uses LAPACK geqrf (full matrix)")
        self.assertFalse(contract.supports_block_matrix, "QR is dense only")
        self.assertFalse(contract.requires_square, "QR works on non-square (thin/reduced QR)")
        
    def test_lu_contract(self):
        """Test LU decomposition contract"""
        contract = self.registry.get_contract("lu")
        self.assertIsNotNone(contract, "lu not registered in OpRegistry")
        self.assertEqual(contract.name, "lu")
        self.assertFalse(contract.supports_streaming, "LU uses LAPACK getrf (full matrix)")
        self.assertFalse(contract.supports_block_matrix, "LU is dense only")
        self.assertTrue(contract.requires_square, "LU requires square matrix")
        
    def test_svd_contract(self):
        """Test SVD contract"""
        contract = self.registry.get_contract("svd")
        self.assertIsNotNone(contract, "svd not registered in OpRegistry")
        self.assertEqual(contract.name, "svd")
        self.assertFalse(contract.supports_streaming, "SVD uses LAPACK gesdd (full matrix)")
        self.assertFalse(contract.supports_block_matrix, "SVD is dense only")
        self.assertFalse(contract.requires_square, "SVD works on non-square matrices")
        
    def test_solve_contract(self):
        """Test linear system solve contract"""
        contract = self.registry.get_contract("solve")
        self.assertIsNotNone(contract, "solve not registered in OpRegistry")
        self.assertEqual(contract.name, "solve")
        self.assertFalse(contract.supports_streaming, "solve uses LAPACK getrf/getrs (full matrix)")
        self.assertFalse(contract.supports_block_matrix, "solve requires dense A")
        self.assertTrue(contract.requires_square, "solve requires square coefficient matrix A")
        
    def test_all_phase7_ops_registered(self):
        """Verify all Phase 7 operations are registered"""
        expected_ops = ["trace", "determinant", "frobenius_norm", "qr", "lu", "svd", "solve"]
        for op_name in expected_ops:
            contract = self.registry.get_contract(op_name)
            self.assertIsNotNone(contract, f"{op_name} should be registered")
            
    def test_streaming_ops_declared_correctly(self):
        """Verify streaming-safe ops are marked correctly"""
        streaming_ops = ["trace", "frobenius_norm"]
        for op_name in streaming_ops:
            contract = self.registry.get_contract(op_name)
            if contract:
                self.assertTrue(
                    contract.supports_streaming,
                    f"{op_name} should support streaming (streaming-safe operation)"
                )
                
    def test_lapack_ops_non_streaming(self):
        """Verify LAPACK-based ops correctly declare no streaming"""
        lapack_ops = ["qr", "lu", "svd", "solve", "determinant"]
        for op_name in lapack_ops:
            contract = self.registry.get_contract(op_name)
            if contract:
                self.assertFalse(
                    contract.supports_streaming,
                    f"{op_name} should not support streaming (LAPACK constraint)"
                )


if __name__ == "__main__":
    unittest.main()
