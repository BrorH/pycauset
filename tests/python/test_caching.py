
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
        import gc
        gc.collect()

        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                print(f"WARNING: Could not clean up {self.test_dir} due to file locking.")

    def test_small_cache_persistence(self):
        m = None
        m2 = None
        try:
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

            # Verify calling trace() returns cached value
            self.assertEqual(m2.trace(), 9.0)
        finally:
            if m is not None:
                m.close()
            if m2 is not None:
                m2.close()

    def test_eigenvalues_persistence(self):
        raise unittest.SkipTest(
            "Eigenvalue caching was removed with the legacy complex subsystem."
        )
        m = None
        m2 = None
        try:
            m = pycauset.FloatMatrix(2)
            m.set(0, 0, 1.0)
            m.set(1, 1, 2.0)

            _ = m.eigenvalues()

            path = self.test_dir / "matrix_evals.pycauset"
            pycauset.save(m, path)

            m2 = pycauset.load(path)

            self.assertTrue(hasattr(m2, "cached_eigenvalues"))
            self.assertEqual(len(m2.cached_eigenvalues), 2)
            v0 = m2.cached_eigenvalues[0]
            self.assertAlmostEqual(v0.real, 1.0)
        finally:
            if m is not None:
                m.close()
            if m2 is not None:
                m2.close()

    def test_eigenvectors_persistence(self):
        raise unittest.SkipTest(
            "Eigenvector caching was removed with the legacy complex subsystem."
        )
        m = None
        m2 = None
        m3 = None
        vecs = None
        vecs3 = None
        try:
            m = pycauset.FloatMatrix(2)
            m.set(0, 0, 1.0)
            m.set(1, 1, 2.0)

            path = self.test_dir / "matrix_evecs.pycauset"
            pycauset.save(m, path)

            # Load and compute eigenvectors with save=True
            m2 = pycauset.load(path)
            vecs = m2.eigenvectors(save=True)

            import zipfile
            with zipfile.ZipFile(path, "r") as zf:
                self.assertIn("eigenvectors.real.bin", zf.namelist())
                self.assertIn("cache.json", zf.namelist())

            # Load again and check if eigenvectors are retrieved from cache
            m3 = pycauset.load(path)
            vecs3 = m3.eigenvectors()

            real_backing = vecs3.real.get_backing_file()
            self.assertEqual(str(Path(real_backing).resolve()), str(path.resolve()))
        finally:
            if vecs3 is not None:
                vecs3.close()
            if vecs is not None:
                vecs.close()
            if m3 is not None:
                m3.close()
            if m2 is not None:
                m2.close()
            if m is not None:
                m.close()

    def test_inverse_persistence(self):
        m = None
        m2 = None
        m3 = None
        inv = None
        inv3 = None
        try:
            m = pycauset.FloatMatrix(2)
            m.set(0, 0, 1.0)
            m.set(1, 1, 2.0)

            path = self.test_dir / "matrix_inv.pycauset"
            pycauset.save(m, path)

            m2 = pycauset.load(path)
            inv = m2.invert(save=True)

            import zipfile
            with zipfile.ZipFile(path, "r") as zf:
                self.assertIn("inverse.bin", zf.namelist())

            m3 = pycauset.load(path)
            inv3 = m3.invert()

            inv_backing = inv3.get_backing_file()
            self.assertEqual(str(Path(inv_backing).resolve()), str(path.resolve()))
        finally:
            if inv3 is not None:
                inv3.close()
            if inv is not None:
                inv.close()
            if m3 is not None:
                m3.close()
            if m2 is not None:
                m2.close()
            if m is not None:
                m.close()

if __name__ == "__main__":
    unittest.main()
