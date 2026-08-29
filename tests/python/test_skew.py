"""Native skew-symmetric eigensolver tests (R2_CATALOG: R2E native skew eigensystem).

`pycauset.eigvals_skew(A, k)` returns the top-k (by magnitude) eigenvalues of a
real skew-symmetric matrix (A == -A.T). Such eigenvalues are purely imaginary
and come in +/-i*lambda pairs (plus a zero eigenvalue for odd dimension).
"""

import unittest

import numpy as np

import pycauset


class TestSkewEigvals(unittest.TestCase):
    def _generate_skew(self, n: int) -> np.ndarray:
        """Return a random real skew-symmetric n x n matrix (A == -A.T)."""
        rng = np.random.default_rng(42)
        M = rng.random((n, n))
        return M - M.T

    def test_topk_matches_numpy(self) -> None:
        """Top-k skew eigenvalues match NumPy's general eigensolver by magnitude."""
        N = 100
        k = 10
        A_np = self._generate_skew(N)
        evals = pycauset.eigvals_skew(pycauset.matrix(A_np), k)

        self.assertEqual(evals.size(), k)

        np_sorted = sorted(np.linalg.eigvals(A_np), key=abs, reverse=True)
        for i in range(k):
            self.assertAlmostEqual(abs(evals.get(i)), abs(np_sorted[i]), places=5)

    def test_purely_imaginary(self) -> None:
        """Skew eigenvalues have a negligible real part."""
        A_np = self._generate_skew(50)
        evals = pycauset.eigvals_skew(pycauset.matrix(A_np), 10)
        max_real = max(abs(evals.get(i).real) for i in range(evals.size()))
        self.assertLess(max_real, 1e-9)

    def test_odd_dimension_has_zero(self) -> None:
        """A skew matrix of odd dimension must have a zero eigenvalue."""
        A_np = self._generate_skew(11)
        evals = pycauset.eigvals_skew(pycauset.matrix(A_np), 11)
        min_mag = min(abs(evals.get(i)) for i in range(evals.size()))
        self.assertLess(min_mag, 1e-9)

    def test_k_clamping(self) -> None:
        """Requesting more eigenvalues than N returns at most N."""
        A_np = self._generate_skew(10)
        evals = pycauset.eigvals_skew(pycauset.matrix(A_np), 20)
        self.assertLessEqual(evals.size(), 10)

    def test_singular_block(self) -> None:
        """A rank-deficient skew matrix returns no more non-zeros than its rank."""
        N = 20
        A_small = self._generate_skew(10)
        A_np = np.zeros((N, N))
        A_np[:10, :10] = A_small
        evals = pycauset.eigvals_skew(pycauset.matrix(A_np), 15)
        non_zeros = sum(1 for i in range(evals.size()) if abs(evals.get(i)) > 1e-5)
        self.assertLessEqual(non_zeros, 10)

    def test_rejects_non_square(self) -> None:
        """Non-square input raises ValueError."""
        with self.assertRaises(ValueError):
            pycauset.eigvals_skew(pycauset.matrix(np.zeros((3, 4))), 2)

    def test_rejects_nonpositive_k(self) -> None:
        """k <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            pycauset.eigvals_skew(pycauset.matrix(self._generate_skew(5)), 0)


class TestSkewEig(unittest.TestCase):
    def _generate_skew(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(7)
        M = rng.random((n, n))
        return M - M.T

    def test_eigenvectors_satisfy_eigequation(self) -> None:
        """Each returned pair satisfies A v = w v."""
        n = 30
        k = 8
        A_np = self._generate_skew(n)
        w, v = pycauset.eig_skew(pycauset.matrix(A_np), k)

        self.assertEqual(w.size(), k)
        self.assertEqual(v.rows(), n)
        self.assertEqual(v.cols(), k)

        for j in range(k):
            vj = np.array([v.get(i, j) for i in range(n)])
            resid = np.linalg.norm(A_np @ vj - w.get(j) * vj)
            self.assertLess(resid, 1e-8)

    def test_topk_by_magnitude(self) -> None:
        """Returned eigenvalues are the top-k by magnitude, matching NumPy."""
        n = 60
        k = 12
        A_np = self._generate_skew(n)
        w, _ = pycauset.eig_skew(pycauset.matrix(A_np), k)

        np_sorted = sorted(np.linalg.eigvals(A_np), key=abs, reverse=True)
        for i in range(k):
            self.assertAlmostEqual(abs(w.get(i)), abs(np_sorted[i]), places=5)

    def test_k_clamping(self) -> None:
        """k larger than n returns n columns."""
        A_np = self._generate_skew(9)
        w, v = pycauset.eig_skew(pycauset.matrix(A_np), 100)
        self.assertEqual(v.cols(), 9)
        self.assertEqual(w.size(), 9)

    def test_rejects_non_square(self) -> None:
        with self.assertRaises(ValueError):
            pycauset.eig_skew(pycauset.matrix(np.zeros((3, 4))), 2)

    def test_rejects_nonpositive_k(self) -> None:
        with self.assertRaises(ValueError):
            pycauset.eig_skew(pycauset.matrix(self._generate_skew(5)), 0)


if __name__ == "__main__":
    unittest.main()
