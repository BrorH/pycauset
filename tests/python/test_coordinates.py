import unittest
import numpy as np
from pycauset import CausalSet, MinkowskiDiamond

class TestCoordinates(unittest.TestCase):
    def test_seed_consistency(self):
        """Test that two CausalSets with the same seed produce identical coordinates."""
        seed = 12345
        n = 50
        c1 = CausalSet(n=n, seed=seed, spacetime=MinkowskiDiamond(2))
        c2 = CausalSet(n=n, seed=seed, spacetime=MinkowskiDiamond(2))
        
        coords1 = c1.coordinates()
        coords2 = c2.coordinates()
        
        np.testing.assert_array_equal(coords1, coords2)

    def test_seed_sensitivity(self):
        """Test that two CausalSets with different seeds produce different coordinates."""
        n = 50
        c1 = CausalSet(n=n, seed=11111, spacetime=MinkowskiDiamond(2))
        c2 = CausalSet(n=n, seed=22222, spacetime=MinkowskiDiamond(2))
        
        coords1 = c1.coordinates()
        coords2 = c2.coordinates()
        
        # It is extremely unlikely that all coordinates are identical
        self.assertFalse(np.array_equal(coords1, coords2))

    def test_dimension_consistency(self):
        """Test that coordinates have the correct dimension."""
        c2 = CausalSet(n=10, spacetime=MinkowskiDiamond(2))
        self.assertEqual(c2.coordinates().shape[1], 2)
        
        c3 = CausalSet(n=10, spacetime=MinkowskiDiamond(3))
        self.assertEqual(c3.coordinates().shape[1], 3)
        
        c4 = CausalSet(n=10, spacetime=MinkowskiDiamond(4))
        self.assertEqual(c4.coordinates().shape[1], 4)

    def test_indices_subset(self):
        """Test that retrieving a subset of indices matches the full set."""
        c = CausalSet(n=100, seed=42, spacetime=MinkowskiDiamond(3))
        all_coords = c.coordinates()
        
        indices = [0, 10, 50, 99]
        sub_coords = c.coordinates(indices=indices)
        
        self.assertEqual(len(sub_coords), 4)
        for i, idx in enumerate(indices):
            np.testing.assert_array_equal(sub_coords[i], all_coords[idx])

if __name__ == '__main__':
    unittest.main()
