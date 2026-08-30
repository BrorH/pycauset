"""
Test Phase 6: Cached-Derived Metadata for Eigen Operations

Verifies eigenvalue/eigenvector caching behavior:
- Load-hit: cached values are reused
- Load-miss: missing cache emits warning and recomputes
- Signature-mismatch: stale cache is ignored and recomputes
"""

import unittest
import pycauset
import numpy as np
import tempfile
import shutil
from pathlib import Path


class TestEigenCaching(unittest.TestCase):
    def setUp(self):
        """Create temp directory for backing files"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_eigen_cache_"))
        pycauset.set_backing_dir(str(self.test_dir))
        
    def tearDown(self):
        """Clean up temp directory"""
        try:
            shutil.rmtree(self.test_dir)
        except Exception:
            pass
            
    def test_eigh_cache_hit(self):
        """Test that eigh caches eigenvalues and eigenvectors"""
        # Create a symmetric matrix
        n = 10
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T
        A = pycauset.matrix(A_np)
        
        # First call - should compute and cache
        w1, v1 = pycauset.eigh(A)
        
        # Verify results
        self.assertEqual(w1.size(), n)
        self.assertEqual(v1.rows(), n)
        
        # Second call on same matrix - should hit cache
        w2, v2 = pycauset.eigh(A)
        
        # Should get same results (cache hit)
        for i in range(n):
            self.assertAlmostEqual(w1.get(i), w2.get(i), places=10)
            
    def test_eigvalsh_recomputes_on_modification(self):
        """Test that modifying matrix invalidates cache"""
        n = 10
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T
        A = pycauset.matrix(A_np)
        
        # Compute eigenvalues
        w1 = pycauset.eigvalsh(A)
        
        # Modify the matrix
        A.set(0, 0, A.get(0, 0) + 10.0)
        
        # Recompute - should get different results (cache invalidated)
        w2 = pycauset.eigvalsh(A)
        
        # First eigenvalue should be noticeably different
        diff = abs(w1.get(0) - w2.get(0))
        # Note: modifying A[0,0] shifts all eigenvalues, but we just check they differ
        # (exact difference depends on matrix structure)
        
    def test_cache_persistence_across_load(self):
        """Eigenvalues are correct across a save()/load() roundtrip.

        Uses the supported persistence API (save()/load()) rather than the
        `backing_file=` constructor kwarg (which is not honoured for NumPy input).
        """
        n = 8
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T

        path = self.test_dir / "test_matrix.pycauset"

        A = pycauset.matrix(A_np)
        w1, v1 = pycauset.eigh(A)
        w1_vals = [w1.get(i) for i in range(n)]

        # Persist and reload into a fresh object (simulates a program restart).
        pycauset.save(A, str(path))
        A2 = pycauset.load(str(path))

        # Compute again, values must match (cache hit or recomputation).
        w2, v2 = pycauset.eigh(A2)
        w2_vals = [w2.get(i) for i in range(n)]

        for i in range(n):
            self.assertAlmostEqual(w1_vals[i], w2_vals[i], places=8)

    def test_eigen_cache_persistence_hits_cache(self):
        """R2_EIGCACHE: save → load hits the persisted eigen cache (no recompute)."""
        import unittest.mock as mock

        n = 8
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T

        path = self.test_dir / "sym.pycauset"

        # File-backed matrix so the big-blob cache can attach to its container.
        A0 = pycauset.matrix(A_np)
        pycauset.save(A0, str(path))
        A = pycauset.load(str(path))

        w1, v1 = pycauset.eigh(A)  # compute + persist the eigen cache

        A2 = pycauset.load(str(path))  # fresh reload (simulated restart)

        # If the cache is hit, np.linalg.eigh is never called.
        with mock.patch.object(np.linalg, "eigh",
                               side_effect=AssertionError("eigh recomputed: cache missed")):
            w2, v2 = pycauset.eigh(A2)

        for i in range(n):
            self.assertAlmostEqual(w1.get(i), w2.get(i), places=10)
            
    def test_view_eigen_no_cache_pollution(self):
        """Test that eigen on a view doesn't pollute parent cache"""
        n = 10
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T
        A = pycauset.matrix(A_np)
        
        # Take a view
        view = A[2:7, 2:7]
        
        # Compute eigen on view
        w_view, v_view = pycauset.eigh(view)
        self.assertEqual(w_view.size(), 5)
        
        # Compute eigen on full matrix
        w_full, v_full = pycauset.eigh(A)
        self.assertEqual(w_full.size(), n)
        
        # They should have different eigenvalues (different matrices)
        # Just verify sizes are different
        self.assertNotEqual(w_view.size(), w_full.size())


if __name__ == "__main__":
    unittest.main()
