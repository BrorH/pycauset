"""R2_DIM — dimension estimators (Myrheim–Meyer, relation fraction)."""

import unittest

import pycauset as pc
from pycauset import synthetic as synth


class TestRelationFraction(unittest.TestCase):
    def test_chain(self):
        self.assertAlmostEqual(synth.chain(30).relation_fraction(), 1.0, places=10)

    def test_antichain(self):
        self.assertEqual(synth.antichain(30).relation_fraction(), 0.0)

    def test_small_n(self):
        self.assertEqual(synth.chain(1).relation_fraction(), 0.0)  # no pairs


class TestMyrheimMeyer(unittest.TestCase):
    def test_recovers_d1_chain(self):
        c = synth.chain(200)
        self.assertAlmostEqual(c.myrheim_meyer_dimension(), 1.0, delta=0.05)

    def test_recovers_d2_diamond(self):
        # A 1+1 Minkowski diamond has relation fraction 1/4 -> Myrheim-Meyer d = 2.
        c = pc.CausalSet(n=2000, spacetime=pc.MinkowskiDiamond(2), seed=3)
        self.assertAlmostEqual(c.myrheim_meyer_dimension(), 2.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
