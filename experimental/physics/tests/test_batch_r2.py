"""R2_BATCH, the `is_causal_batch` fast path matches the element-wise path exactly."""

import unittest

import numpy as np
import pycauset as pc
from pycauset import spacetime as sp


class TestBatchHook(unittest.TestCase):
    def test_batch_matches_elementwise(self):
        rng = np.random.default_rng(0)
        for st in (sp.MinkowskiDiamond(2), sp.MinkowskiBox(3, 2.0, 3.0),
                   sp.MinkowskiCylinder(2, 2.0, 3.0)):
            coords = st.sample(rng, 40)
            batch = st.is_causal_batch(coords)
            self.assertEqual(batch.shape, (40, 40))
            for i in range(40):
                for j in range(40):
                    self.assertEqual(bool(batch[i, j]), st.is_causal(coords[i], coords[j]))

    def test_sprinkler_uses_batch_consistently(self):
        # The matrix built by the (batch) sprinkler must equal the element-wise
        # causal relation on the time-sorted coordinates.
        st = sp.MinkowskiBox(3, 2.0, 3.0)
        c = pc.CausalSet(n=60, spacetime=st, seed=7)
        C = np.asarray(c.C, dtype=bool)
        coords = c.coordinates()  # time-sorted
        batch = st.is_causal_batch(coords)
        np.testing.assert_array_equal(C, np.triu(batch, 1))

    def test_fallback_matches_batch(self):
        # A spacetime with the batch hook disabled (element-wise fallback) must give
        # the same order as the batch path for the same seed.
        st = sp.MinkowskiBox(3, 2.0, 3.0)
        c_batch = pc.CausalSet(n=60, spacetime=st, seed=7)

        # Force the fallback by shadowing the batch hook with one that raises.
        class _NoBatch(sp.MinkowskiBox):
            def is_causal_batch(self, coords):
                raise NotImplementedError

        st2 = _NoBatch(3, 2.0, 3.0)
        c_fallback = pc.CausalSet(n=60, spacetime=st2, seed=7)
        np.testing.assert_array_equal(
            np.asarray(c_batch.C, dtype=bool), np.asarray(c_fallback.C, dtype=bool)
        )


if __name__ == "__main__":
    unittest.main()
