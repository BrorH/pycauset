import unittest
import numpy as np
import pycauset
from pycauset import CausalSet, MinkowskiDiamond

# Mock plotly if not installed
import sys
from unittest.mock import MagicMock

try:
    import plotly
except ImportError:
    # Mock plotly for testing in CI/environments without it
    sys.modules["plotly"] = MagicMock()
    sys.modules["plotly.graph_objects"] = MagicMock()
    sys.modules["plotly.express"] = MagicMock()

from pycauset.vis import plot_embedding

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.c = CausalSet(n=100, spacetime=MinkowskiDiamond(2))
        self.c_large = CausalSet(n=1000, spacetime=MinkowskiDiamond(3))

    def test_coordinates_retrieval(self):
        # Test getting all coordinates
        coords = self.c.coordinates()
        self.assertEqual(coords.shape, (100, 2))
        
        # Test getting subset
        indices = [0, 5, 99]
        coords_sub = self.c.coordinates(indices=indices)
        self.assertEqual(coords_sub.shape, (3, 2))
        
        # Verify consistency (deterministic)
        coords_all = self.c.coordinates()
        self.assertTrue(np.allclose(coords_sub[0], coords_all[0]))
        self.assertTrue(np.allclose(coords_sub[2], coords_all[99]))

    def test_coordinates_safety(self):
        # Create a small causet to avoid expensive sprinkling
        c_huge = CausalSet(n=100, spacetime=MinkowskiDiamond(2))
        # Hack the size to simulate a large set
        c_huge._n = 200000
        
        # Should warn if trying to get all
        with self.assertRaises(UserWarning):
            c_huge.coordinates()
            
        # Should work with force=True
        # We don't actually run this because make_coordinates might try to generate 200k points
        # even if we hacked _n, passing the hacked _n to C++ might cause it to try to generate that many.
        # So we just test the guard clause.
        pass

    def test_plot_embedding_2d(self):
        # Should return a Figure object (or mock)
        fig = plot_embedding(self.c)
        self.assertIsNotNone(fig)

    def test_plot_embedding_3d(self):
        fig = plot_embedding(self.c_large)
        self.assertIsNotNone(fig)
        
    def test_sampling(self):
        # Test that sampling works for "large" sets
        # We set sample_size small to force sampling
        fig = plot_embedding(self.c, sample_size=10)
        self.assertIsNotNone(fig)

if __name__ == '__main__':
    unittest.main()
