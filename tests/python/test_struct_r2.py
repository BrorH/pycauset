"""R2_STRUCT, causal-structure methods: links, chains/antichains, intervals, layering."""

import unittest

import numpy as np
import pycauset as pc
from pycauset import CausalSet


def _poset(edges):
    """Build a CausalSet from a list of (i, j) covering/order edges."""
    n = max(max(e) for e in edges) + 1
    C = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        C[i, j] = True
    # take the transitive closure (only reflexive-free upper-triangular kept)
    C2 = C.copy()
    for k in range(n):
        C2 |= (C2[:, k, None] & C2[None, k, :])
    C2 = np.triu(C2, 1)
    return CausalSet(n=n, matrix=pc.causal_matrix(C2))


class TestLinks(unittest.TestCase):
    def test_diamond_links(self):
        # Diamond poset: 0 < 1 < 3 and 0 < 2 < 3 (so 0 < 3 is NOT a link).
        c = _poset([(0, 1), (0, 2), (1, 3), (2, 3)])
        L = c.links()
        self.assertTrue(L[0, 1] and L[0, 2] and L[1, 3] and L[2, 3])
        self.assertFalse(L[0, 3])  # covered by a length-2 path

    def test_links_matches_formula(self):
        c = CausalSet(n=60, spacetime=pc.MinkowskiDiamond(2), seed=5)
        C = np.asarray(c.C, dtype=bool)
        L_ref = C & ~((C.astype(np.uint8) @ C.astype(np.uint8)) > 0)
        np.testing.assert_array_equal(c.links(), L_ref)


class TestPastFutureInterval(unittest.TestCase):
    def setUp(self):
        self.c = _poset([(0, 1), (0, 2), (1, 3), (2, 3)])

    def test_past(self):
        self.assertEqual(set(self.c.past(3)), {0, 1, 2})
        self.assertEqual(set(self.c.past(1)), {0})

    def test_future(self):
        self.assertEqual(set(self.c.future(0)), {1, 2, 3})
        self.assertEqual(set(self.c.future(2)), {3})

    def test_interval(self):
        # I(0, 3) = future(0) ∩ past(3) = {1, 2, 3} ∩ {0, 1, 2} = {1, 2}
        self.assertEqual(set(self.c.interval(0, 3)), {1, 2})


class TestChainsAntichains(unittest.TestCase):
    def test_longest_chain_of_total_order(self):
        c = _poset([(0, 1), (1, 2), (2, 3)])
        chain = c.longest_chain()
        self.assertEqual(len(chain), 4)
        self.assertTrue(c.is_chain(chain))

    def test_longest_chain_diamond(self):
        c = _poset([(0, 1), (0, 2), (1, 3), (2, 3)])
        self.assertEqual(len(c.longest_chain()), 3)

    def test_chain_and_antichain_predicates(self):
        c = _poset([(0, 1), (0, 2), (1, 3), (2, 3)])
        self.assertTrue(c.is_chain([0, 1, 3]))
        self.assertFalse(c.is_chain([1, 2]))      # 1 and 2 are incomparable
        self.assertTrue(c.is_antichain([1, 2]))
        self.assertFalse(c.is_antichain([0, 1]))  # comparable

    def test_layers(self):
        c = _poset([(0, 1), (0, 2), (1, 3), (2, 3)])
        layers = c.layers()
        # layer 0 = {0}, layer 1 = {1, 2}, layer 2 = {3}
        self.assertEqual([set(l) for l in layers], [{0}, {1, 2}, {3}])


if __name__ == "__main__":
    unittest.main()
