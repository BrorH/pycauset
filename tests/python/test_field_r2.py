"""R2_FIELD / R2_KRD / R2_SJ, Field → CorrelatedField, propagators, and SJ Wightman."""

import unittest

import numpy as np
import pycauset as pc


class TestFieldModel(unittest.TestCase):
    def setUp(self):
        self.c = pc.CausalSet(n=30, spacetime=pc.MinkowskiDiamond(2), seed=3)

    def test_field_factory(self):
        phi = pc.field("scalar", mass=1.5)
        self.assertIsInstance(phi, pc.Field)
        self.assertEqual(phi.mass, 1.5)
        self.assertEqual(phi.kind, "scalar")

    def test_unknown_kind_raises(self):
        with self.assertRaises(NotImplementedError):
            pc.field("fermion", mass=1.0)

    def test_on_returns_correlated_field(self):
        phi = pc.field("scalar", mass=1.0)
        Q = phi.on(self.c)
        self.assertIsInstance(Q, pc.CorrelatedField)
        self.assertIs(Q.causet, self.c)
        self.assertEqual(Q.mass, 1.0)

    def test_backcompat_scalarfield_import(self):
        from pycauset.field import ScalarField
        f = ScalarField(self.c, mass=1.0)
        self.assertAlmostEqual(f.mass, 1.0)


class TestPropagators(unittest.TestCase):
    def setUp(self):
        self.c = pc.CausalSet(n=30, spacetime=pc.MinkowskiDiamond(2), seed=3)

    def test_retarded_matches_formula(self):
        c = pc.CausalSet(n=20, spacetime=pc.MinkowskiDiamond(2), seed=1)
        Q = pc.field("scalar", mass=1.0).on(c)

        C = np.asarray(c.C, dtype=float)
        n = c.n
        rho = c.density  # 20 / 1.0
        a = 0.5
        b = -(1.0 ** 2) / rho

        K_ref = a * C @ np.linalg.inv(np.eye(n) - b * a * C)
        np.testing.assert_allclose(Q.retarded(), K_ref, atol=1e-10)

    def test_advanced_is_transpose(self):
        Q = pc.field("scalar", mass=1.0).on(self.c)
        np.testing.assert_allclose(Q.advanced(), Q.retarded().T, atol=1e-12)

    def test_massless_limit_is_aC(self):
        # R2_CATALOG shortcut: b -> 0 gives K_R = aC (no solve at all).
        c = pc.CausalSet(n=10, spacetime=pc.MinkowskiDiamond(2), seed=1)
        Q = pc.field("scalar", mass=0.0).on(c)
        C = np.asarray(c.C, dtype=float)
        np.testing.assert_allclose(Q.retarded(), 0.5 * C, atol=1e-12)

    def test_pauli_jordan_is_hermitian(self):
        Q = pc.field("scalar", mass=1.0).on(self.c)
        iD = Q.pauli_jordan()
        np.testing.assert_allclose(iD, iD.conj().T, atol=1e-10)

    def test_pauli_jordan_is_kr_minus_ka(self):
        Q = pc.field("scalar", mass=1.0).on(self.c)
        iD = Q.pauli_jordan()
        np.testing.assert_allclose(iD, 1j * (Q.retarded() - Q.advanced()), atol=1e-10)


class TestWightman(unittest.TestCase):
    def setUp(self):
        self.Q = pc.field("scalar", mass=1.0).on(
            pc.CausalSet(n=30, spacetime=pc.MinkowskiDiamond(2), seed=3)
        )

    def test_wightman_is_positive_part(self):
        W = self.Q.wightman()
        iD = self.Q.pauli_jordan()

        # W must be Hermitian and positive-semidefinite.
        np.testing.assert_allclose(W, W.conj().T, atol=1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(W).min(), -1e-10)

        # W equals (iΔ + |iΔ|) / 2, the positive-eigenvalue part of iΔ.
        evals, evecs = np.linalg.eigh(iD)
        abs_iD = (evecs * np.abs(evals)) @ evecs.conj().T
        np.testing.assert_allclose(W, (iD + abs_iD) / 2, atol=1e-8)

    def test_wightman_reconstructs_pauli_jordan(self):
        # iΔ = W + N where N is the negative-eigenvalue part.
        W = self.Q.wightman()
        iD = self.Q.pauli_jordan()
        evals, evecs = np.linalg.eigh(iD)
        N = (evecs * np.clip(evals, None, 0.0)) @ evecs.conj().T
        np.testing.assert_allclose(W + N, iD, atol=1e-8)

    def test_correlator_is_wightman(self):
        np.testing.assert_allclose(self.Q.correlator(), self.Q.wightman(), atol=0.0)


if __name__ == "__main__":
    unittest.main()
