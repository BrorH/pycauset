"""
Test Phase 6: Eigen Op Contract Registration

Verifies that all eigen operations are properly registered in the OpRegistry
with correct capabilities declared.
"""

import unittest

try:
    import _pycauset as internal
except ImportError:
    try:
        from pycauset import _pycauset as internal
    except ImportError:
        internal = None


class TestEigenOpContract(unittest.TestCase):
    def setUp(self):
        if internal is None:
            self.skipTest("_pycauset module not available")
        self.registry = internal.OpRegistry.instance()
        
    def test_eigh_contract(self):
        """Test eigh operation contract"""
        contract = self.registry.get_contract("eigh")
        self.assertIsNotNone(contract, "eigh not registered in OpRegistry")
        self.assertEqual(contract.name, "eigh")
        self.assertFalse(contract.supports_streaming, "eigh should not support streaming (LAPACK full matrix)")
        self.assertFalse(contract.supports_block_matrix, "eigh should not support block matrices")
        self.assertTrue(contract.requires_square, "eigh requires square matrices")
        
    def test_eigvalsh_contract(self):
        """Test eigvalsh operation contract"""
        contract = self.registry.get_contract("eigvalsh")
        self.assertIsNotNone(contract, "eigvalsh not registered in OpRegistry")
        self.assertEqual(contract.name, "eigvalsh")
        self.assertFalse(contract.supports_streaming, "eigvalsh should not support streaming")
        self.assertFalse(contract.supports_block_matrix, "eigvalsh should not support block matrices")
        self.assertTrue(contract.requires_square, "eigvalsh requires square matrices")
        
    def test_eig_contract(self):
        """Test eig (general) operation contract"""
        contract = self.registry.get_contract("eig")
        self.assertIsNotNone(contract, "eig not registered in OpRegistry")
        self.assertEqual(contract.name, "eig")
        self.assertFalse(contract.supports_streaming, "eig should not support streaming")
        self.assertFalse(contract.supports_block_matrix, "eig should not support block matrices")
        self.assertTrue(contract.requires_square, "eig requires square matrices")
        
    def test_eigvals_contract(self):
        """Test eigvals (general) operation contract"""
        contract = self.registry.get_contract("eigvals")
        self.assertIsNotNone(contract, "eigvals not registered in OpRegistry")
        self.assertEqual(contract.name, "eigvals")
        self.assertFalse(contract.supports_streaming, "eigvals should not support streaming")
        self.assertFalse(contract.supports_block_matrix, "eigvals should not support block matrices")
        self.assertTrue(contract.requires_square, "eigvals requires square matrices")
        
    def test_eigvals_arnoldi_contract(self):
        """Test eigvals_arnoldi operation contract"""
        contract = self.registry.get_contract("eigvals_arnoldi")
        self.assertIsNotNone(contract, "eigvals_arnoldi not registered in OpRegistry")
        self.assertEqual(contract.name, "eigvals_arnoldi")
        self.assertTrue(contract.supports_streaming, "Arnoldi should support streaming (matrix-vector products)")
        self.assertFalse(contract.supports_block_matrix, "eigvals_arnoldi currently dense only")
        self.assertTrue(contract.requires_square, "eigvals_arnoldi requires square matrices")
        
    def test_all_eigen_ops_registered(self):
        """Verify all eigen operations are registered"""
        expected_ops = ["eigh", "eigvalsh", "eig", "eigvals", "eigvals_arnoldi"]
        for op_name in expected_ops:
            contract = self.registry.get_contract(op_name)
            self.assertIsNotNone(contract, f"{op_name} should be registered")


if __name__ == "__main__":
    unittest.main()
