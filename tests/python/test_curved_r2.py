"""R2_CURVED — DeSitter / AntiDeSitter / FLRW (documented parametrizations)."""

import unittest

import numpy as np
import pycauset as pc
from pycauset import spacetime as sp


class TestDeSitter(unittest.TestCase):
    def test_sprinkles_to_valid_order(self):
        c = pc.CausalSet(n=40, spacetime=sp.DeSitter(2), seed=3)
        c.validate()

    def test_is_causal(self):
        st = sp.DeSitter(2)
        # same spatial direction, future time -> timelike future
        self.assertTrue(st.is_causal([0.0, 0.0], [0.5, 0.0]))
        # past -> not causal
        self.assertFalse(st.is_causal([0.5, 0.0], [0.0, 0.0]))
        # opposite spatial direction, small time -> spacelike
        self.assertFalse(st.is_causal([0.0, 0.0], [0.1, np.pi]))

    def test_scalar_coeffs_raises(self):
        with self.assertRaises(NotImplementedError):
            sp.DeSitter(2).scalar_coeffs(1.0, 10.0)


class TestAntiDeSitter(unittest.TestCase):
    def test_no_causal_order(self):
        st = sp.AntiDeSitter(2)
        with self.assertRaises(NotImplementedError):
            st.is_causal([0.0, 0.0], [0.1, 0.1])

    def test_sample_shape(self):
        rng = np.random.default_rng(0)
        self.assertEqual(sp.AntiDeSitter(3).sample(rng, 5).shape, (5, 3))


class TestFLRW(unittest.TestCase):
    def test_flat_is_minkowski_like(self):
        st = sp.FLRW(2, scale_factor=0, time_extent=5.0, space_extent=5.0)
        self.assertTrue(st.is_causal([0.0, 0.0], [2.0, 1.0]))
        self.assertFalse(st.is_causal([0.0, 0.0], [1.0, 2.0]))

    def test_sprinkles_to_valid_order(self):
        c = pc.CausalSet(n=40, spacetime=sp.FLRW(2, scale_factor=0), seed=5)
        c.validate()

    def test_scalar_coeffs_raises(self):
        with self.assertRaises(NotImplementedError):
            sp.FLRW(2).scalar_coeffs(1.0, 10.0)


if __name__ == "__main__":
    unittest.main()
