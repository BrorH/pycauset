"""R2_ENT — Sorkin–Yazdi entanglement entropy (two documented conventions)."""

import unittest

import numpy as np
import pycauset as pc


class TestEntanglementEntropy(unittest.TestCase):
    def setUp(self):
        self.c = pc.CausalSet(n=60, spacetime=pc.MinkowskiDiamond(2), seed=3)
        self.Q = pc.field("scalar", mass=1.0).on(self.c)
        self.region = list(range(20))

    def test_nonnegative(self):
        S = self.Q.entanglement_entropy(self.region)
        self.assertGreaterEqual(S, -1e-9)

    def test_empty_region_zero(self):
        self.assertEqual(self.Q.entanglement_entropy([]), 0.0)

    def test_nonzero_for_entangled_region(self):
        S = self.Q.entanglement_entropy(self.region)
        self.assertGreater(S, 0.0)

    def test_symplectic_raises_for_sj_wightman(self):
        # The SJ Wightman has eigenvalues < 1/2, so the literal symplectic form
        # is undefined; it must raise rather than silently produce NaN.
        with self.assertRaises(ValueError):
            self.Q.entanglement_entropy(self.region, convention="symplectic")

    def test_unknown_convention_raises(self):
        with self.assertRaises(ValueError):
            self.Q.entanglement_entropy(self.region, convention="bogus")

    def test_conventions_agree_after_shift(self):
        # sorkin_yazdi(W) == symplectic(W + 1/2 I) — the two conventions are the same
        # formula up to the zero-point shift.
        W = self.Q.wightman()
        idx = self.region
        W_A = W[np.ix_(idx, idx)]

        S_sorkin = self.Q.entanglement_entropy(idx, convention="sorkin_yazdi")

        W_shift = W_A + 0.5 * np.eye(len(idx))
        evals = np.linalg.eigvalsh(W_shift)
        with np.errstate(divide="ignore", invalid="ignore"):
            term2 = (evals - 0.5) * np.log(evals - 0.5)
        term2 = np.where(evals - 0.5 > 0.0, term2, 0.0)  # 0 ln 0 = 0
        S_symp = float(np.sum((evals + 0.5) * np.log(evals + 0.5) - term2))

        self.assertAlmostEqual(S_sorkin, S_symp, places=8)


if __name__ == "__main__":
    unittest.main()
