"""R2_FIELD (State) + R2_CORR (vevs) — the Field → CorrelatedField → State layer."""

import unittest

import numpy as np
import pycauset as pc


class TestState(unittest.TestCase):
    def setUp(self):
        self.c = pc.CausalSet(n=30, spacetime=pc.MinkowskiDiamond(2), seed=3)
        self.Q = pc.field("scalar", mass=1.0).on(self.c)

    def test_vacuum_state(self):
        st = self.Q.state()
        self.assertIsInstance(st, pc.State)
        np.testing.assert_allclose(st.field(), np.zeros(self.c.n))
        # vacuum 2-point = Wightman W
        np.testing.assert_allclose(st.two_point(), self.Q.wightman(), atol=1e-10)

    def test_config_state(self):
        config = np.arange(self.c.n, dtype=float)
        st = self.Q.state(config)
        np.testing.assert_allclose(st.field(), config)

    def test_two_point(self):
        config = np.arange(self.c.n, dtype=float)
        st = self.Q.state(config)
        W = self.Q.wightman()
        np.testing.assert_allclose(st.two_point(), np.outer(config, config) + W, atol=1e-10)

    def test_field_variance(self):
        config = np.arange(self.c.n, dtype=float)
        st = self.Q.state(config)
        W = self.Q.wightman()
        np.testing.assert_allclose(st.field_variance(), np.real(np.diag(W)) + config ** 2, atol=1e-10)

    def test_bad_config_shape(self):
        with self.assertRaises(ValueError):
            self.Q.state(np.zeros(5))


if __name__ == "__main__":
    unittest.main()
