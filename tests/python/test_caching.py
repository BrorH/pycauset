
import unittest
import os
import shutil
import numpy as np
import pycauset
from pathlib import Path

class TestCaching(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_caching_output")
        self.test_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_small_cache_persistence(self):
        # Create a matrix
        m = pycauset.FloatMatrix(3)
        m.set(0, 0, 2.0)
        m.set(1, 1, 3.0)
        m.set(2, 2, 4.0)
        
        # Compute trace and determinant
        tr = m.trace()
        det = m.determinant()
        
        self.assertEqual(tr, 9.0)
        self.assertEqual(det, 24.0)
        
        # Save
        path = self.test_dir / "matrix_cache.pycauset"
        pycauset.save(m, path)
        
        # Load
        m2 = pycauset.load(path)
        
        # Check if cached values are present
        self.assertTrue(hasattr(m2, "cached_trace"))
        self.assertEqual(m2.cached_trace, 9.0)
        
        self.assertTrue(hasattr(m2, "cached_determinant"))
        self.assertEqual(m2.cached_determinant, 24.0)
        
        # Verify calling trace() returns cached value (we can't easily verify it didn't recompute, 
        # but we can verify it returns correct value)
        self.assertEqual(m2.trace(), 9.0)
        
        m.close()
        m2.close()

    def test_eigenvalues_persistence(self):
        m = pycauset.FloatMatrix(2)
        m.set(0, 0, 1.0)
        m.set(1, 1, 2.0)
        
        evals = m.eigenvalues()
        
        path = self.test_dir / "matrix_evals.pycauset"
        pycauset.save(m, path)
        
        m2 = pycauset.load(path)
        
        self.assertTrue(hasattr(m2, "cached_eigenvalues"))
        self.assertEqual(len(m2.cached_eigenvalues), 2)
        # Check values
        v0 = m2.cached_eigenvalues[0]
        self.assertAlmostEqual(v0.real, 1.0)
        
        m.close()
        m2.close()

    def test_eigenvectors_persistence(self):
        m = pycauset.FloatMatrix(2)
        m.set(0, 0, 1.0)
        m.set(1, 1, 2.0)
        
        path = self.test_dir / "matrix_evecs.pycauset"
        pycauset.save(m, path)
        
        # Load and compute eigenvectors with save=True
        m2 = pycauset.load(path)
        vecs = m2.eigenvectors(save=True)
        
        # Check if files exist in ZIP
        import zipfile
        with zipfile.ZipFile(path, "r") as zf:
            self.assertIn("eigenvectors.real.bin", zf.namelist())
            self.assertIn("cache.json", zf.namelist())
            
        # Load again and check if eigenvectors are retrieved from cache
        m3 = pycauset.load(path)
        # We can't easily check internal cache state from Python without peeking private members
        # But we can check if calling eigenvectors() returns a ComplexMatrix backed by the ZIP
        
        vecs3 = m3.eigenvectors()
        
        # Check backing file of vecs3
        real_backing = vecs3.real.get_backing_file()
        self.assertEqual(str(Path(real_backing).resolve()), str(path.resolve()))
        
        m.close()
        m2.close()
        m3.close()
        vecs.close()
        vecs3.close()

    def test_inverse_persistence(self):
        m = pycauset.FloatMatrix(2)
        m.set(0, 0, 1.0)
        m.set(1, 1, 2.0)
        
        path = self.test_dir / "matrix_inv.pycauset"
        pycauset.save(m, path)
        
        m2 = pycauset.load(path)
        inv = m2.invert(save=True)
        
        # Check ZIP
        import zipfile
        with zipfile.ZipFile(path, "r") as zf:
            self.assertIn("inverse.bin", zf.namelist())
            
        m3 = pycauset.load(path)
        inv3 = m3.invert()
        
        # Check backing
        inv_backing = inv3.get_backing_file()
        self.assertEqual(str(Path(inv_backing).resolve()), str(path.resolve()))
        
        m.close()
        m2.close()
        m3.close()
        inv.close()
        inv3.close()

if __name__ == "__main__":
    unittest.main()
