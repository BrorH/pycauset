"""R2_BH — Schwarzschild (1+1 radial, exact tortoise null condition)."""

import unittest

import numpy as np
import pycauset as pc
from pycauset import spacetime as sp
from pycauset.spacetime import _tortoise


class TestSchwarzschild(unittest.TestCase):
    def test_radial_causality(self):
        st = sp.Schwarzschild(mass=1.0)
        # outgoing timelike: large dt, increasing r
        self.assertTrue(st.is_causal([0.0, 5.0], [5.0, 6.0]))
        # past -> not causal
        self.assertFalse(st.is_causal([5.0, 6.0], [0.0, 5.0]))
        # too-short dt for the radial separation -> not causal
        self.assertFalse(st.is_causal([0.0, 5.0], [0.1, 9.0]))

    def test_tortoise_matches_null_condition(self):
        st = sp.Schwarzschild(mass=1.0)
        r1, r2 = 5.0, 7.0
        dt_null = abs(float(_tortoise(r2, 1.0) - _tortoise(r1, 1.0)))
        self.assertTrue(st.is_causal([0.0, r1], [dt_null + 0.01, r2]))
        self.assertFalse(st.is_causal([0.0, r1], [dt_null - 0.01, r2]))

    def test_sprinkles_to_valid_order(self):
        c = pc.CausalSet(n=30, spacetime=sp.Schwarzschild(mass=1.0), seed=1)
        c.validate()

    def test_higher_dimension_raises(self):
        with self.assertRaises(NotImplementedError):
            sp.Schwarzschild(dimension=3)

    def test_scalar_coeffs_raises(self):
        with self.assertRaises(NotImplementedError):
            sp.Schwarzschild(mass=1.0).scalar_coeffs(1.0, 10.0)


if __name__ == "__main__":
    unittest.main()
