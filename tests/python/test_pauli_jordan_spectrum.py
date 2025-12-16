import unittest

raise unittest.SkipTest(
    "Matrix eigenvalues/eigendecomposition APIs were removed with the legacy complex subsystem."
)

import pycauset
import numpy as np
import os

class TestPauliJordanSpectrum(unittest.TestCase):
    def setUp(self):
        # Create a small causal set
        self.st = pycauset.spacetime.MinkowskiDiamond(2)
        self.c = pycauset.CausalSet(density=50, spacetime=self.st)
        self.field = pycauset.field.ScalarField(self.c, mass=1.0)

    def tearDown(self):
        # Clean up any temporary files if needed
        pass

    def test_spectrum_symmetry(self):
        """
        Test that the spectrum of the Pauli-Jordan function i*Delta
        is symmetric (pairs of +/- lambda) and has even rank.
        """
        # Compute i*Delta
        # This returns an AntiSymmetricMatrix with scalar=1j
        Delta = self.field.pauli_jordan()
        
        # Verify it is antisymmetric and has complex scalar
        self.assertTrue(Delta.is_antisymmetric())
        self.assertEqual(Delta.scalar, 1j)
        
        # Compute eigenvalues
        # Since Delta is AntiSymmetric real * 1j, it is Hermitian.
        # So eigenvalues should be real.
        evals_complex = Delta.eigenvalues()
        
        # Convert to numpy array for easier analysis
        evals = np.array(evals_complex)
        
        # Check that eigenvalues are mostly real (imaginary part should be negligible)
        # i * (AntiSymmetric Real) -> Hermitian -> Real Eigenvalues
        max_imag = np.max(np.abs(evals.imag))
        self.assertLess(max_imag, 1e-10, "Eigenvalues of i*Delta should be real")
        
        # Work with real parts
        evals_real = evals.real
        
        # Filter out zeros (numerical noise)
        tolerance = 1e-8
        non_zero_evals = evals_real[np.abs(evals_real) > tolerance]
        
        # Check rank is even
        rank = len(non_zero_evals)
        print(f"Rank of Delta: {rank}")
        self.assertTrue(rank % 2 == 0, f"Rank should be even, got {rank}")
        
        # Check for pairs +/- lambda
        # Sort by absolute value
        sorted_evals = np.sort(non_zero_evals)
        
        # If we have pairs, the sum should be close to 0
        # (Since for every x there is -x)
        sum_evals = np.sum(non_zero_evals)
        self.assertLess(np.abs(sum_evals), 1e-8, "Sum of non-zero eigenvalues should be 0")
        
        # More rigorous check: sort positive and negative halves and compare
        pos_evals = np.sort(non_zero_evals[non_zero_evals > 0])
        neg_evals = np.sort(np.abs(non_zero_evals[non_zero_evals < 0]))
        
        self.assertEqual(len(pos_evals), len(neg_evals), "Should have equal number of positive and negative eigenvalues")
        
        # Check difference
        diff = np.abs(pos_evals - neg_evals)
        max_diff = np.max(diff)
        self.assertLess(max_diff, 1e-8, "Positive and negative eigenvalues should match in magnitude")

if __name__ == '__main__':
    unittest.main()
