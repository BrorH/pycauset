"""Regression pins for known correctness bugs.

Each still-broken path is marked ``unittest.expectedFailure``; when fixed it
reports as an *unexpected success*, signalling the marker can be removed.
"""

import unittest

import numpy as np

import pycauset as pc
import pycauset._pycauset as native


class TestKnownBugs(unittest.TestCase):
    def test_solve_returns_correct_solution(self):
        # FIXED 2026-08-24: native solve/lu returned unique_ptr<MatrixBase>,
        # which pybind11 mishandled (dangling downcast). Now returned as shared_ptr.
        rng = np.random.default_rng(0)
        a = rng.random((5, 5))
        a += np.eye(5) * 5
        b = rng.random((5, 2))
        x = np.array(pc.solve(pc.matrix(a), pc.matrix(b)))
        np.testing.assert_allclose(a @ x, b, atol=1e-8)

    def test_lu_completes_and_reconstructs(self):
        a = np.random.default_rng(0).random((5, 5))
        p, l, u = pc.lu(pc.matrix(a))
        rec = np.array(p) @ np.array(l) @ np.array(u)
        np.testing.assert_allclose(rec, a, atol=1e-8)

    @unittest.expectedFailure
    def test_triangular_bit_matrix_random_size(self):
        # BUG (open): TriangularBitMatrix.random(5) reports .size() == 25 (n*n)
        # instead of 5.
        tbm = native.TriangularBitMatrix.random(5, p=0.5)
        self.assertEqual(tbm.size(), 5)


if __name__ == "__main__":
    unittest.main()
