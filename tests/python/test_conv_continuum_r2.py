"""R2_CONV / R2_CMVP, pin the iΔ convention against the continuum (massless 1+1).

For the massless 1+1 field the discrete Pauli–Jordan function is **exact**: the
discrete ``iΔ = (i/2)(C − Cᵀ)`` equals the continuum ``iΔ = (i/2) sgn(Δt) θ(σ)``
sampled at the causet's points. This pins the sign/scale convention with no
approximation.
"""

import unittest

import numpy as np
import pycauset as pc


class TestContinuumComparison(unittest.TestCase):
    def setUp(self):
        self.st = pc.spacetime.MinkowskiDiamond(2)
        self.c = pc.CausalSet(n=60, spacetime=self.st, seed=11)

    def test_on_spacetime_returns_continuum(self):
        Q_ct = pc.field("scalar", mass=0.0).on(self.st)
        self.assertIsInstance(Q_ct, pc.ContinuumCorrelatedField)

    def test_discrete_pauli_jordan_matches_continuum(self):
        # Discrete iΔ vs continuum iΔ sampled at the (physical) points. The
        # diamond samples in lightcone (u,v); `to_embedding` gives physical (t,x).
        Q_c = pc.field("scalar", mass=0.0).on(self.c)
        Q_ct = pc.field("scalar", mass=0.0).on(self.st)

        coords = self.st.to_embedding(self.c.coordinates())
        iD_discrete = Q_c.pauli_jordan()
        iD_continuum = Q_ct.at(coords, which="pauli_jordan")

        np.testing.assert_allclose(iD_discrete, iD_continuum, atol=1e-12)

    def test_continuum_retarded_matches_discrete_kr(self):
        Q_c = pc.field("scalar", mass=0.0).on(self.c)
        Q_ct = pc.field("scalar", mass=0.0).on(self.st)

        coords = self.st.to_embedding(self.c.coordinates())
        KR_discrete = Q_c.retarded()
        GR_continuum = Q_ct.at(coords, which="retarded")

        np.testing.assert_allclose(KR_discrete, GR_continuum, atol=1e-12)

    def test_continuum_pauli_jordan_spot_checks(self):
        Q_ct = pc.field("scalar", mass=0.0).on(self.st)
        # future timelike: +i/2 ; past timelike: -i/2 ; spacelike: 0
        self.assertEqual(Q_ct.pauli_jordan([0.0, 0.0], [1.0, 0.5]), 0.5j)
        self.assertEqual(Q_ct.pauli_jordan([1.0, 0.5], [0.0, 0.0]), -0.5j)
        self.assertEqual(Q_ct.pauli_jordan([0.0, 0.0], [0.5, 1.0]), 0j)


if __name__ == "__main__":
    unittest.main()
