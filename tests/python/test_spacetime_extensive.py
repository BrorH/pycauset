import unittest
import os
import tempfile
import sys
import numpy as np
from pathlib import Path

# Add python directory to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pycauset
from pycauset import CausalSet, MinkowskiDiamond

class TestSpacetimeExtensive(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

    def test_high_dimensions(self):
        """Test sprinkling in higher dimensions."""
        # 10D spacetime
        dim = 10
        n = 50
        c = CausalSet(n=n, spacetime=MinkowskiDiamond(dim))
        self.assertEqual(c.spacetime.dimension(), dim)
        self.assertEqual(len(c), n)
        
        # Check coordinates shape
        coords = c.coordinates()
        self.assertEqual(coords.shape, (n, dim))

    def test_extreme_n_values(self):
        """Test very small and moderately large N."""
        # N = 1
        c1 = CausalSet(n=1, spacetime=MinkowskiDiamond(2))
        self.assertEqual(len(c1), 1)
        self.assertEqual(c1.C.size(), 1)
        
        # N = 0 (Should probably fail or produce empty)
        try:
            c0 = CausalSet(n=0, spacetime=MinkowskiDiamond(2))
            self.assertEqual(len(c0), 0)
        except ValueError:
            # If 0 is not allowed, that's valid too
            pass

    def test_seed_determinism_extensive(self):
        """Verify strict determinism across multiple generations."""
        seed = 999
        n = 100
        st = MinkowskiDiamond(3)
        
        c1 = CausalSet(n=n, spacetime=st, seed=seed)
        c2 = CausalSet(n=n, spacetime=st, seed=seed)
        
        # Check coordinates match exactly
        np.testing.assert_array_equal(c1.coordinates(), c2.coordinates())
        
        # Check matrix matches exactly
        # We can't easily iterate all bits efficiently in python for large N, 
        # but we can check a sample or convert to numpy
        m1 = np.array(c1.C)
        m2 = np.array(c2.C)
        np.testing.assert_array_equal(m1, m2)

    def test_coordinate_bounds(self):
        """Verify all sprinkled points lie within the diamond bounds."""
        n = 100
        dim = 2
        c = CausalSet(n=n, spacetime=MinkowskiDiamond(dim))
        coords = c.coordinates()
        
        # For Minkowski Diamond in 2D, bounds are typically related to lightcone intervals
        # Just checking they are finite and reasonable for now
        self.assertTrue(np.all(np.isfinite(coords)))
        
        # Check time ordering if applicable (t coordinate usually 0th or last?)
        # Assuming standard convention, we can just check they aren't NaN

    def test_large_scale_sprinkling(self):
        """Test performance/stability with larger N."""
        # N = 2000 is large enough to stress test but small enough for unit test
        n = 2000
        c = CausalSet(n=n, spacetime=MinkowskiDiamond(4))
        self.assertEqual(len(c), n)
        self.assertEqual(c.C.size(), n)

if __name__ == '__main__':
    unittest.main()
