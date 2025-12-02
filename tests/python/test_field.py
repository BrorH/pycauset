import unittest
import math
import pycauset
from pycauset import CausalSet
from pycauset.field import ScalarField

class TestScalarField(unittest.TestCase):
    def setUp(self):
        # Create a small 2D causal set for testing
        self.c2d = CausalSet(n=100, density=100.0, spacetime=pycauset.MinkowskiDiamond(2))
        # Create a small 4D causal set for testing
        self.c4d = CausalSet(n=100, density=100.0, spacetime=pycauset.MinkowskiDiamond(4))

    def test_initialization(self):
        field = ScalarField(self.c2d, mass=1.5)
        self.assertEqual(field.mass, 1.5)
        self.assertEqual(field.causet, self.c2d)
        
        field_default = ScalarField(self.c2d)
        self.assertEqual(field_default.mass, 0.0)

    def test_coeffs_2d(self):
        field = ScalarField(self.c2d, mass=2.0)
        # Access private method for testing purposes
        a, b = field._get_coeffs()
        
        # Expected for 2D: a = 0.5, b = -m^2 / rho
        expected_a = 0.5
        expected_b = - (2.0**2) / self.c2d.density
        
        self.assertAlmostEqual(a, expected_a)
        self.assertAlmostEqual(b, expected_b)

    def test_coeffs_4d(self):
        field = ScalarField(self.c4d, mass=2.0)
        a, b = field._get_coeffs()
        
        # Expected for 4D: a = sqrt(rho) / (2 * pi * sqrt(6)), b = -m^2 / rho
        rho = self.c4d.density
        expected_a = math.sqrt(rho) / (2 * math.pi * math.sqrt(6))
        expected_b = - (2.0**2) / rho
        
        self.assertAlmostEqual(a, expected_a)
        self.assertAlmostEqual(b, expected_b)

    def test_massless_propagator(self):
        # Massless field (b=0)
        field = ScalarField(self.c2d, mass=0.0)
        K = field.propagator()
        
        # Should be a * C
        # For 2D, a = 0.5
        C = self.c2d.C
        expected_K = 0.5 * C
        
        # Check a few elements
        self.assertAlmostEqual(K[0, 0], expected_K[0, 0])
        if self.c2d.n > 1:
            self.assertAlmostEqual(K[1, 0], expected_K[1, 0])

    def test_massive_propagator_structure(self):
        field = ScalarField(self.c2d, mass=1.0)
        K = field.propagator()
        
        # Check dimensions
        self.assertEqual(K.shape[0], self.c2d.n)
        self.assertEqual(K.shape[1], self.c2d.n)
        
        # Check type (should be float matrix)
        self.assertTrue("Float" in str(K))

    def test_manual_coeffs(self):
        field = ScalarField(self.c2d, mass=1.0)
        # Override a and b
        K = field.propagator(a=1.0, b=-0.5)
        
        # With a=1, b=-0.5
        # alpha_eff = -1 / (1 * -0.5) = 2.0
        # K = (-1/-0.5) * compute_k(C, 2.0) = 2.0 * compute_k(C, 2.0)
        
        from pycauset import compute_k
        expected_K = 2.0 * compute_k(self.c2d.C, 2.0)
        
        # Compare matrices (simple check of first element)
        self.assertAlmostEqual(K[0, 0], expected_K[0, 0])

    def test_missing_density_error(self):
        # Create a causal set without density info (simulated by manually setting n but not density)
        # Note: CausalSet constructor requires n or density. If we pass n, density is calculated if spacetime has volume.
        # We need to mock a situation where density is unknown.
        
        # Let's try to create a dummy object that looks like a CausalSet but fails on density
        class DummyCauset:
            def __init__(self):
                self.density = None
                self.C = None 
                
                class MinkowskiDiamond: # Name matches what we expect
                    def dimension(self): return 2
                
                self.spacetime = MinkowskiDiamond()
                
        dummy = DummyCauset()
        field = ScalarField(dummy, mass=1.0)
        
        with self.assertRaises(ValueError):
            field._get_coeffs()

    def test_unsupported_dimension_error(self):
        # 3D is not supported yet
        c3d = CausalSet(n=100, density=100.0, spacetime=pycauset.MinkowskiDiamond(3))
        field = ScalarField(c3d, mass=1.0)
        
        with self.assertRaises(NotImplementedError):
            field._get_coeffs()

if __name__ == '__main__':
    unittest.main()
