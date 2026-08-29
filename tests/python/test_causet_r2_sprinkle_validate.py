"""R2_ABC (sprinkler) + R2_VALIDATE, custom Spacetime sprinkling and order validation."""

import unittest

import numpy as np
import pycauset
from pycauset import CausalSet
from pycauset import spacetime as sp
from pycauset.causet import validate_causal_matrix


class _Diamond2D(sp.Spacetime):
    """A pure-Python 1+1 Minkowski diamond (mirrors the native MinkowskiDiamond)."""

    def dimension(self):
        return 2

    def volume(self):
        return 1.0

    def sample(self, rng, n):
        return rng.uniform(0.0, 1.0, size=(n, 2))

    def is_causal(self, u, v):
        return u[0] < v[0] and u[1] < v[1]


class TestValidateCausalMatrix(unittest.TestCase):
    def test_valid_matrix_passes(self):
        # The transitive closure of a 4-chain (every i < j is related).
        A = np.array(
            [[0, 1, 1, 1],
             [0, 0, 1, 1],
             [0, 0, 0, 1],
             [0, 0, 0, 0]],
            dtype=bool,
        )
        validate_causal_matrix(A)  # no raise

    def test_reflexive_rejected(self):
        A = np.eye(3, dtype=bool)
        with self.assertRaises(ValueError):
            validate_causal_matrix(A)

    def test_antisymmetry_rejected(self):
        A = np.array([[0, 1], [1, 0]], dtype=bool)
        with self.assertRaises(ValueError):
            validate_causal_matrix(A)

    def test_transitivity_rejected(self):
        # 0 -> 1 -> 2, but no direct 0 -> 2.
        A = np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [0, 0, 0]],
            dtype=bool,
        )
        with self.assertRaises(ValueError):
            validate_causal_matrix(A)

    def test_non_square_rejected(self):
        with self.assertRaises(ValueError):
            validate_causal_matrix(np.zeros((2, 3), dtype=bool))


class TestCustomSpacetimeSprinkle(unittest.TestCase):
    def test_sprinkles_to_valid_order(self):
        c = CausalSet(n=50, spacetime=_Diamond2D(), seed=42)
        self.assertEqual(c.n, 50)
        self.assertEqual(c.C.rows(), 50)
        self.assertEqual(c.C.cols(), 50)
        c.validate()  # must be a valid partial order

    def test_attaches_embedding(self):
        c = CausalSet(n=30, spacetime=_Diamond2D(), seed=7)
        self.assertIsNotNone(c.embedding)
        self.assertEqual(c.embedding.shape, (30, 2))
        # coordinates() serves the attached embedding
        coords = c.coordinates()
        np.testing.assert_array_equal(coords, c.embedding)
        # subset
        sub = c.coordinates(indices=[0, 5, 10])
        np.testing.assert_array_equal(sub, c.embedding[[0, 5, 10]])

    def test_reproducible_with_seed(self):
        c1 = CausalSet(n=40, spacetime=_Diamond2D(), seed=123)
        c2 = CausalSet(n=40, spacetime=_Diamond2D(), seed=123)
        np.testing.assert_array_equal(
            np.asarray(c1.C, dtype=bool), np.asarray(c2.C, dtype=bool)
        )

    def test_euclidean_spacetime_refuses_causal_order(self):
        class Euclidean(sp.Spacetime):
            signature = (0, 2)

            def dimension(self):
                return 2

            def volume(self):
                return 1.0

            def sample(self, rng, n):
                return rng.uniform(0.0, 1.0, size=(n, 2))

        with self.assertRaises(NotImplementedError):
            CausalSet(n=10, spacetime=Euclidean(), seed=1)


class TestCausalSetValidation(unittest.TestCase):
    def _nontransitive_matrix(self):
        C = np.zeros((3, 3), dtype=bool)
        C[0, 1] = True
        C[1, 2] = True  # path 0 -> 1 -> 2, but 0 -> 2 is missing
        return pycauset.causal_matrix(C)

    def test_matrix_eager_validation(self):
        m = self._nontransitive_matrix()
        with self.assertRaises(ValueError):
            CausalSet(n=3, matrix=m)

    def test_matrix_validate_false_escape(self):
        m = self._nontransitive_matrix()
        c = CausalSet(n=3, matrix=m, validate=False)
        self.assertEqual(c.n, 3)

    def test_validate_method_raises(self):
        m = self._nontransitive_matrix()
        c = CausalSet(n=3, matrix=m, validate=False)
        with self.assertRaises(ValueError):
            c.validate()


if __name__ == "__main__":
    unittest.main()
