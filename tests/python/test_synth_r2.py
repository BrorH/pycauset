"""R2_SYNTH — synthetic poset generators (Chain, Antichain, percolation, products, Poset)."""

import unittest

import numpy as np
import pycauset as pc
from pycauset import synthetic as synth


class TestSyntheticGenerators(unittest.TestCase):
    def test_chain(self):
        c = synth.chain(5)
        self.assertEqual(c.n, 5)
        c.validate()
        self.assertTrue(np.all(np.asarray(c.C, dtype=bool)[np.triu_indices(5, 1)]))

    def test_antichain(self):
        c = synth.antichain(5)
        c.validate()
        self.assertFalse(np.asarray(c.C, dtype=bool).any())

    def test_transitive_percolation(self):
        c = synth.transitive_percolation(0.4, 20, seed=1)
        c.validate()  # must be a valid order
        self.assertEqual(c.n, 20)

    def test_random_dag_order(self):
        c = synth.random_dag_order(0.4, 20, seed=2)
        c.validate()

    def test_product_order(self):
        c = synth.product_order((2, 3))  # grid poset of 6 elements
        self.assertEqual(c.n, 6)
        c.validate()

    def test_poset(self):
        c = synth.poset([(0, 1), (1, 2)])  # a 3-chain
        self.assertEqual(c.n, 3)
        c.validate()
        self.assertEqual(len(c.longest_chain()), 3)

    def test_reproducible_with_seed(self):
        a = synth.transitive_percolation(0.4, 20, seed=7)
        b = synth.transitive_percolation(0.4, 20, seed=7)
        np.testing.assert_array_equal(np.asarray(a.C, dtype=bool), np.asarray(b.C, dtype=bool))


if __name__ == "__main__":
    unittest.main()
