"""R2_MINK, the flat Minkowski family: correct order, volume, sampler, causality."""

import unittest

import numpy as np
from pycauset import CausalSet
from pycauset import spacetime as sp


class TestMinkowskiOrder(unittest.TestCase):
    def test_diamond_order_is_transitive(self):
        for d in (2, 3, 4):
            c = CausalSet(n=60, spacetime=sp.MinkowskiDiamond(d), seed=42)
            c.validate()  # reflexive-free, antisymmetric, transitive

    def test_box_order_is_transitive(self):
        c = CausalSet(n=60, spacetime=sp.MinkowskiBox(3, 2.0, 3.0), seed=7)
        c.validate()

    def test_cylinder_order_is_transitive(self):
        c = CausalSet(n=60, spacetime=sp.MinkowskiCylinder(2, 2.0, 3.0), seed=9)
        c.validate()

    def test_reproducibility(self):
        a = CausalSet(n=80, spacetime=sp.MinkowskiBox(3, 2.0, 1.0), seed=123)
        b = CausalSet(n=80, spacetime=sp.MinkowskiBox(3, 2.0, 1.0), seed=123)
        np.testing.assert_array_equal(np.asarray(a.C, dtype=bool), np.asarray(b.C, dtype=bool))


class TestMinkowskiVolume(unittest.TestCase):
    def test_diamond_volume(self):
        self.assertAlmostEqual(sp.MinkowskiDiamond(2).volume(), 1.0)

    def test_box_volume(self):
        self.assertAlmostEqual(sp.MinkowskiBox(2, 2.0, 1.0).volume(), 2.0)
        self.assertAlmostEqual(sp.MinkowskiBox(3, 2.0, 3.0).volume(), 18.0)

    def test_cylinder_volume(self):
        self.assertAlmostEqual(sp.MinkowskiCylinder(2, 2.0, 3.0).volume(), 6.0)

    def test_diamond_sampler_bounds(self):
        rng = np.random.default_rng(0)
        pts = sp.MinkowskiDiamond(4).sample(rng, 5000)
        self.assertTrue(np.all(pts >= 0.0) and np.all(pts <= 1.0))

    def test_box_sampler_bounds(self):
        rng = np.random.default_rng(0)
        pts = sp.MinkowskiBox(3, 2.0, 3.0).sample(rng, 5000)
        self.assertTrue(np.all(pts[:, 0] >= 0.0) and np.all(pts[:, 0] <= 2.0))
        self.assertTrue(np.all(pts[:, 1:] >= 0.0) and np.all(pts[:, 1:] <= 3.0))


class TestMinkowskiCausality(unittest.TestCase):
    def test_diamond_causality(self):
        st = sp.MinkowskiDiamond(2)
        self.assertTrue(st.is_causal([0.1, 0.1], [0.5, 0.5]))
        self.assertFalse(st.is_causal([0.5, 0.1], [0.2, 0.5]))
        self.assertFalse(st.is_causal([0.1, 0.5], [0.5, 0.1]))

    def test_box_causality(self):
        st = sp.MinkowskiBox(2, 10.0, 10.0)
        self.assertFalse(st.is_causal([0.0, 0.0], [1.0, 1.0]))  # lightlike
        self.assertTrue(st.is_causal([0.0, 0.0], [2.0, 1.0]))  # timelike
        self.assertFalse(st.is_causal([0.0, 0.0], [1.0, 2.0]))  # spacelike

    def test_cylinder_causality(self):
        st = sp.MinkowskiCylinder(2, 10.0, 10.0)
        self.assertTrue(st.is_causal([0.0, 0.0], [1.0, 0.5]))
        self.assertFalse(st.is_causal([0.0, 0.0], [1.0, 2.0]))
        # wrap-around: x=0.0 vs x=9.0 are distance 1.0 apart on a circle of C=10
        self.assertFalse(st.is_causal([0.0, 0.0], [0.1, 9.0]))  # dt=0.1 < dx=1.0


class TestMinkowskiErrors(unittest.TestCase):
    def test_cylinder_dimension_error(self):
        with self.assertRaises(NotImplementedError):
            sp.MinkowskiCylinder(3, 1.0, 1.0)

    def test_diamond_dimension_error(self):
        with self.assertRaises(ValueError):
            sp.MinkowskiDiamond(1)


if __name__ == "__main__":
    unittest.main()
