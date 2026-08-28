"""Everyday-use scenario tests.

Each scenario mirrors a realistic causal-set workflow and is implemented in at
least two different ways. The ways must agree with each other and with a NumPy
reference computed from the raw causal matrix.
"""

import unittest

import numpy as np

import pycauset as pc


def _dense(mat, n):
    """Materialize a causal (TriangularBitMatrix) matrix to a dense int ndarray."""
    out = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            out[i, j] = 1 if mat.get(i, j) else 0
    return out


def _causal(n, *, seed, dim=3):
    """Build one causal set in a diamond and return its causal matrix C."""
    return pc.CausalSet(n, spacetime=pc.MinkowskiDiamond(dim), seed=seed).C


class ScenarioCubeAndSum(unittest.TestCase):
    """Make 3 causal sets in a 3D diamond; cube all three causal matrices and add them."""

    def setUp(self):
        self.n = 16
        self.C = [_causal(self.n, seed=1000 + k) for k in (1, 2, 3)]
        self.ref = [np.linalg.matrix_power(_dense(c, self.n), 3) for c in self.C]

    def _want(self):
        return self.ref[0] + self.ref[1] + self.ref[2]

    def test_way_operator_chain(self):
        C = self.C
        result = (C[0] @ C[0]) @ C[0] + (C[1] @ C[1]) @ C[1] + (C[2] @ C[2]) @ C[2]
        np.testing.assert_allclose(pc.to_numpy(result), self._want())

    def test_way_matrix_power(self):
        C = self.C
        result = pc.matrix_power(C[0], 3) + pc.matrix_power(C[1], 3) + pc.matrix_power(C[2], 3)
        np.testing.assert_allclose(pc.to_numpy(result), self._want())

    def test_ways_agree(self):
        C = self.C
        a = (C[0] @ C[0]) @ C[0] + (C[1] @ C[1]) @ C[1] + (C[2] @ C[2]) @ C[2]
        b = pc.matrix_power(C[0], 3) + pc.matrix_power(C[1], 3) + pc.matrix_power(C[2], 3)
        np.testing.assert_allclose(pc.to_numpy(a), pc.to_numpy(b))

    def test_matrix_power_stays_triangular(self):
        # Powers of a triangular causal matrix should stay triangular, not densify.
        self.assertIsInstance(pc.matrix_power(self.C[0], 2), pc.TriangularIntegerMatrix)
        self.assertIsInstance(pc.matrix_power(self.C[0], 4), pc.TriangularIntegerMatrix)


class ScenarioTwoPathCounts(unittest.TestCase):
    """Count the number of 2-step causal relations between every pair (C^2)."""

    def setUp(self):
        self.n = 20
        self.C = _causal(self.n, seed=42)
        self.Cn = _dense(self.C, self.n)

    def test_way_operator(self):
        np.testing.assert_allclose(pc.to_numpy(self.C @ self.C), self.Cn @ self.Cn)

    def test_way_matmul_function(self):
        np.testing.assert_allclose(pc.to_numpy(pc.matmul(self.C, self.C)), self.Cn @ self.Cn)

    def test_way_dot_function(self):
        np.testing.assert_allclose(pc.to_numpy(pc.dot(self.C, self.C)), self.Cn @ self.Cn)

    def test_ways_agree(self):
        a = self.C @ self.C
        b = pc.matmul(self.C, self.C)
        c = pc.dot(self.C, self.C)
        np.testing.assert_allclose(pc.to_numpy(a), pc.to_numpy(b))
        np.testing.assert_allclose(pc.to_numpy(a), pc.to_numpy(c))


class ScenarioFutureCounts(unittest.TestCase):
    """Count how many elements lie in the causal future of each element (row sums of C)."""

    def setUp(self):
        self.n = 18
        self.C = _causal(self.n, seed=7)
        self.Cn = _dense(self.C, self.n)

    def test_way_matvec(self):
        ones = pc.ones(self.n, dtype="int32")
        future = self.C @ ones
        np.testing.assert_allclose(
            np.ravel(pc.to_numpy(future)), self.Cn @ np.ones(self.n, dtype=int)
        )

    def test_future_counts_float64_and_bool_matvec(self):
        want = self.Cn @ np.ones(self.n, dtype=int)
        ones_f = pc.ones(self.n, dtype="float64")
        ones_b = pc.ones(self.n, dtype="bool")
        np.testing.assert_allclose(np.ravel(pc.to_numpy(self.C @ ones_f)), want)
        np.testing.assert_allclose(np.ravel(pc.to_numpy(self.C @ ones_b)), want)

    def test_way_numpy_row_sums(self):
        # The same counts via a direct NumPy reduction over the dense form.
        np.testing.assert_allclose(self.Cn.sum(axis=1), self.Cn @ np.ones(self.n, dtype=int))


class ScenarioLazyAllocation(unittest.TestCase):
    """Allocate a constant working matrix without a dtype, then use it."""

    def test_lazy_vs_explicit_bool_matmul(self):
        n = 12
        C = _causal(n, seed=3)
        lazy = pc.ones((n, n))                    # no dtype: resolves on use
        explicit = pc.ones((n, n), dtype=pc.bool_)
        np.testing.assert_allclose(pc.to_numpy(lazy @ C), pc.to_numpy(explicit @ C))

    def test_lazy_zeros_fill_then_read(self):
        z = pc.zeros((4, 4))
        z.fill(5)                                  # first write deduces int32
        self.assertEqual(z.dtype, "int32")
        np.testing.assert_array_equal(pc.to_numpy(z), np.full((4, 4), 5, dtype=np.int32))


class ScenarioKMatrix(unittest.TestCase):
    """Compute K = C (I + C)^-1, a standard causal-set construction."""

    def test_compute_k_matches_numpy(self):
        n = 14
        C = _causal(n, seed=11)
        Cn = _dense(C, n).astype(float)
        want = Cn @ np.linalg.inv(np.eye(n) + Cn)
        got = pc.to_numpy(pc.compute_k(C, 1.0))
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
