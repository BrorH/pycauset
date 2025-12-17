import unittest
import os
import shutil
import tempfile
import pycauset
from pycauset import CausalSet, MinkowskiDiamond

class TestSpacetimeStructure(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_sprinkling_by_n(self):
        n = 50
        c = CausalSet(n=n, spacetime=MinkowskiDiamond(2))
        self.assertEqual(c.n, n)
        self.assertEqual(len(c), n)
        self.assertIsNotNone(c.C)
        self.assertEqual(c.C.rows(), n)
        self.assertEqual(c.C.cols(), n)
        # Density should be calculated
        self.assertIsNotNone(c.density)
        self.assertAlmostEqual(c.density, n / c.spacetime.volume())

    def test_sprinkling_by_density(self):
        density = 100.0
        st = MinkowskiDiamond(2)
        c = CausalSet(density=density, spacetime=st, seed=42)
        
        # n is Poisson distributed, so it might not be exactly density * volume
        # But it should be deterministic with a seed
        expected_n = c.n
        
        # Re-run with same seed
        c2 = CausalSet(density=density, spacetime=st, seed=42)
        self.assertEqual(c2.n, expected_n)
        self.assertEqual(c.n, c2.n)
        
        # Check density property (it returns actual density n/V)
        # We can't check equality with input density, but we can check it's close
        # or just check that it's consistent
        self.assertAlmostEqual(c.density, c.n / st.volume())

    def test_spacetime_dimensions(self):
        # Test 2D, 3D, 4D
        dims = [2, 3, 4]
        for d in dims:
            c = CausalSet(n=20, spacetime=MinkowskiDiamond(d))
            self.assertEqual(c.spacetime.dimension(), d)
            self.assertEqual(c.C.rows(), 20)
            self.assertEqual(c.C.cols(), 20)

    def test_persistence(self):
        # Create a causet
        c = CausalSet(n=50, spacetime=MinkowskiDiamond(2), seed=123)
        file_path = os.path.join(self.test_dir, "test_causet.causet")
        
        # Save
        c.save(file_path)
        self.assertTrue(os.path.exists(file_path))
        
        # Load
        c_loaded = pycauset.load(file_path)
        
        # Verify
        self.assertEqual(c.n, c_loaded.n)
        self.assertEqual(c.spacetime.dimension(), c_loaded.spacetime.dimension())
        # Matrices should be identical
        # (Assuming we can compare matrices, or at least their properties)
        self.assertEqual(c.C.rows(), c_loaded.C.rows())
        self.assertEqual(c.C.cols(), c_loaded.C.cols())
        
        # Check a few elements to ensure matrix content is preserved
        # (This depends on matrix implementation, assuming [] access works)
        # Since it's a bit matrix, we can check a few bits
        for i in range(min(5, c.n)):
            for j in range(min(5, c.n)):
                self.assertEqual(c.C[i, j], c_loaded.C[i, j])

    def test_matrix_properties(self):
        c = CausalSet(n=10, spacetime=MinkowskiDiamond(2))
        matrix = c.C
        
        # Check it's a TriangularBitMatrix (or whatever the default is)
        self.assertTrue("BitMatrix" in str(type(matrix)) or "BitMatrix" in str(matrix))
        
        # Check size
        self.assertEqual(matrix.shape[0], 10)
        self.assertEqual(matrix.shape[1], 10)

    def test_invalid_init(self):
        # Must provide n or density
        with self.assertRaises(ValueError):
            CausalSet(spacetime=MinkowskiDiamond(2))

if __name__ == '__main__':
    unittest.main()
