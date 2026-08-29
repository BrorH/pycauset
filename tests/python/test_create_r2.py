"""R2_CREATE — composition decorators and `export_python` code generation."""

import unittest

import numpy as np
import pycauset as pc
from pycauset import spacetime as sp


class TestExportPython(unittest.TestCase):
    def test_recipe_codegen_roundtrip(self):
        recipe = {"name": "MyBox", "dimension": 2, "domain": "box",
                  "time_extent": 2.0, "space_extent": 1.0}
        code = sp.export_python(recipe)
        self.assertIn("class MyBox", code)
        self.assertIn("spacetime.create", code)

        ns = {"spacetime": sp}
        exec(code, ns)
        st = ns["MyBox"]()
        self.assertEqual(st.dimension(), 2)
        self.assertEqual(st.volume(), 2.0)

        c = pc.CausalSet(n=40, spacetime=st, seed=1)
        c.validate()  # the generated spacetime sprinkles to a valid order

    def test_spacetime_codegen(self):
        code = sp.export_python(sp.MinkowskiBox(2, 2.0, 1.0))
        self.assertIn("domain='box'", code)
        self.assertIn("dimension=2", code)

    def test_export_python_requires_dimension(self):
        with self.assertRaises(ValueError):
            sp.export_python({"domain": "diamond"})


class TestRestrictedSpacetime(unittest.TestCase):
    def test_restriction(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        st = sp.RestrictedSpacetime(base, region=lambda c: c[1] < 5.0)

        # Volume is (approximately) half the box = 50.0
        self.assertAlmostEqual(st.volume(), 50.0, delta=2.0)

        rng = np.random.default_rng(0)
        pts = st.sample(rng, 200)
        self.assertEqual(pts.shape, (200, 2))
        self.assertTrue(np.all(pts[:, 1] < 5.0))

        # is_causal is inherited from the base
        self.assertEqual(st.is_causal([0.0, 0.0], [1.0, 0.5]),
                         base.is_causal([0.0, 0.0], [1.0, 0.5]))


class TestTransformedSpacetime(unittest.TestCase):
    def test_translation_preserves_contract(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        shift = np.array([0.0, 3.0])
        st = sp.TransformedSpacetime(
            base,
            forward=lambda c: c + shift,
            inverse=lambda c: c - shift,
        )

        self.assertEqual(st.dimension(), 2)
        self.assertEqual(st.volume(), base.volume())  # translation is volume-preserving
        # causality is translation-invariant: pull back by the inverse shift
        self.assertEqual(st.is_causal([0.0, 3.0], [1.0, 3.5]),
                         base.is_causal([0.0, 0.0], [1.0, 0.5]))

        rng = np.random.default_rng(0)
        pts = st.sample(rng, 50)
        self.assertEqual(pts.shape, (50, 2))


class TestConformalSpacetime(unittest.TestCase):
    def test_constant_factor_scales_volume(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)  # volume = 100
        conf = sp.ConformalSpacetime(base, conformal_factor=lambda c: 2.0)
        # Omega^d = 2^2 = 4 -> volume ~= 400 (Monte Carlo estimate)
        self.assertAlmostEqual(conf.volume(), 400.0, delta=40.0)

    def test_causality_is_inherited(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        conf = sp.ConformalSpacetime(base, conformal_factor=lambda c: 1.5)
        for u, v in [([0, 0], [1, 0.5]), ([0, 0], [5, 9]), ([2, 1], [4, 0])]:
            self.assertEqual(conf.is_causal(u, v), base.is_causal(u, v))

    def test_signature_and_dimension_delegated(self):
        base = sp.MinkowskiBox(3, 2.0, 1.0)
        conf = sp.ConformalSpacetime(base, conformal_factor=lambda c: 1.0)
        self.assertEqual(conf.dimension(), 3)
        self.assertEqual(conf.signature, base.signature)

    def test_explicit_volume_overrides_estimate(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        conf = sp.ConformalSpacetime(base, conformal_factor=lambda c: 2.0, volume=123.0)
        self.assertEqual(conf.volume(), 123.0)

    def test_sample_shape(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        conf = sp.ConformalSpacetime(base, conformal_factor=lambda c: 2.0)
        rng = np.random.default_rng(0)
        pts = conf.sample(rng, 50)
        self.assertEqual(pts.shape, (50, 2))


class TestPeriodicSpacetime(unittest.TestCase):
    def test_sample_wraps_to_fundamental_domain(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        per = sp.PeriodicSpacetime(base, periods={1: 5.0})
        rng = np.random.default_rng(0)
        pts = per.sample(rng, 300)
        self.assertTrue(np.all(pts[:, 1] >= 0.0))
        self.assertTrue(np.all(pts[:, 1] < 5.0))

    def test_volume_is_base_volume(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        per = sp.PeriodicSpacetime(base, periods={1: 5.0})
        self.assertEqual(per.volume(), base.volume())

    def test_causal_order_uses_periodic_images(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        per = sp.PeriodicSpacetime(base, periods={1: 10.0})
        # (0,0) -> (5,9) is NOT causal in the base (5 < 9), but the image
        # (5, -1) is causal (5 > 1).
        self.assertFalse(base.is_causal([0.0, 0.0], [5.0, 9.0]))
        self.assertTrue(per.is_causal([0.0, 0.0], [5.0, 9.0]))
        # a direct relation still holds
        self.assertTrue(per.is_causal([0.0, 0.0], [5.0, 1.0]))

    def test_periodic_time_raises(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        with self.assertRaises(NotImplementedError):
            sp.PeriodicSpacetime(base, periods={0: 2.0})

    def test_bad_axis_raises(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        with self.assertRaises(ValueError):
            sp.PeriodicSpacetime(base, periods={5: 2.0})

    def test_bad_period_raises(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        with self.assertRaises(ValueError):
            sp.PeriodicSpacetime(base, periods={1: -1.0})

    def test_bare_number_wraps_all_spatial_axes(self):
        base = sp.MinkowskiBox(3, 4.0, 2.0)
        per = sp.PeriodicSpacetime(base, periods=3.0)
        self.assertEqual(per._periods, {1: 3.0, 2: 3.0})
        rng = np.random.default_rng(0)
        pts = per.sample(rng, 200)
        self.assertTrue(np.all(pts[:, 1:] >= 0.0))
        self.assertTrue(np.all(pts[:, 1:] < 3.0))

    def test_full_causet_validates(self):
        base = sp.MinkowskiBox(2, 10.0, 10.0)
        per = sp.PeriodicSpacetime(base, periods={1: 5.0})
        cs = pc.CausalSet(n=60, spacetime=per, seed=7)
        cs.validate()  # transitivity holds under periodic identification


if __name__ == "__main__":
    unittest.main()
