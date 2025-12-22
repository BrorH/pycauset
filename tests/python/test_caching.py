
import unittest
import os
import shutil
import warnings
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

            # Cached-derived values are surfaced via obj.properties.
            self.assertTrue(hasattr(m2, "properties"))
            self.assertEqual(m2.properties.get("trace"), 9.0)
            self.assertEqual(m2.properties.get("determinant"), 24.0)

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

    def test_eigenvectors_persistence(self):
        raise unittest.SkipTest(
            "Eigenvector caching was removed with the legacy complex subsystem."
        )

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

            # Returns a correct inverse matrix.
            self.assertAlmostEqual(inv.get(0, 0), 1.0)
            self.assertAlmostEqual(inv.get(1, 1), 0.5)

            # Load again and verify inverse is retrieved from the file-backed cache.
            m3 = pycauset.load(path)
            inv3 = m3.invert()

            self.assertAlmostEqual(inv3.get(0, 0), 1.0)
            self.assertAlmostEqual(inv3.get(1, 1), 0.5)

            store_dir = path.with_suffix(path.suffix + ".objects")
            cache_files = list(store_dir.glob("*.pycauset"))
            self.assertEqual(len(cache_files), 1)

            inv3_backing = Path(inv3.get_backing_file()).resolve()
            self.assertEqual(str(inv3_backing), str(cache_files[0].resolve()))
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

    def test_inverse_cache_missing_warns_and_recomputes(self):
        m = None
        m2 = None
        m3 = None
        inv = None
        inv3 = None
        try:
            m = pycauset.FloatMatrix(2)
            m.set(0, 0, 1.0)
            m.set(1, 1, 2.0)

            path = self.test_dir / "matrix_inv_missing_cache.pycauset"
            pycauset.save(m, path)

            m2 = pycauset.load(path)
            inv = m2.invert(save=True)
            inv.close()
            inv = None
            m2.close()
            m2 = None

            store_dir = path.with_suffix(path.suffix + ".objects")
            cache_files = list(store_dir.glob("*.pycauset"))
            self.assertEqual(len(cache_files), 1)
            cache_files[0].unlink()

            m3 = pycauset.load(path)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                inv3 = m3.invert()

            self.assertTrue(any(issubclass(x.category, pycauset.PyCausetStorageWarning) for x in w))
            self.assertAlmostEqual(inv3.get(0, 0), 1.0)
            self.assertAlmostEqual(inv3.get(1, 1), 0.5)
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
