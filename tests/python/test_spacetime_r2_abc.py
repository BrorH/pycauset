"""R2_SIG + R2_ABC — Spacetime ABC, signature model, and registry.

Exercises the `pycauset.spacetime` extension seam against the real package.
"""

import unittest

import pycauset
from pycauset import spacetime as sp


class _Lorentzian(sp.Spacetime):
    def dimension(self):
        return 2

    def volume(self):
        return 1.0

    def sample(self, rng, n):
        return rng.uniform(0.0, 1.0, size=(n, 2))

    def is_causal(self, u, v):
        return u[0] < v[0]


class _Euclidean(sp.Spacetime):
    signature = (0, 2)

    def dimension(self):
        return 2

    def volume(self):
        return 1.0

    def sample(self, rng, n):
        return rng.uniform(0.0, 1.0, size=(n, 2))


class _MultiTime(sp.Spacetime):
    signature = (2, 1)

    def dimension(self):
        return 3

    def volume(self):
        return 1.0

    def sample(self, rng, n):
        return rng.uniform(0.0, 1.0, size=(n, 3))


class TestSpacetimeABC(unittest.TestCase):
    def test_spacetime_is_abstract(self):
        with self.assertRaises(TypeError):
            sp.Spacetime()

    def test_missing_abstract_methods_rejected(self):
        class Bad(sp.Spacetime):
            def dimension(self):
                return 2
            # volume() and sample() are missing

        with self.assertRaises(TypeError):
            Bad()

    def test_minimal_subclass_is_concrete(self):
        class Minimal(sp.Spacetime):
            def dimension(self):
                return 2

            def volume(self):
                return 1.0

            def sample(self, rng, n):
                return rng.uniform(0.0, 1.0, size=(n, 2))

        inst = Minimal()
        self.assertEqual(inst.dimension(), 2)
        self.assertEqual(inst.signature, (1, 1))

    def test_signature_default_lorentzian(self):
        self.assertEqual(_Lorentzian().signature, (1, 1))

    def test_signature_override(self):
        self.assertEqual(_Euclidean().signature, (0, 2))
        self.assertEqual(_MultiTime().signature, (2, 1))

    def test_is_causal_lorentzian_override_works(self):
        self.assertTrue(_Lorentzian().is_causal([0.0, 0.0], [1.0, 1.0]))

    def test_is_causal_raises_euclidean(self):
        with self.assertRaises(NotImplementedError):
            _Euclidean().is_causal([0.0, 0.0], [1.0, 1.0])

    def test_is_causal_raises_multitime(self):
        with self.assertRaises(NotImplementedError):
            _MultiTime().is_causal([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    def test_is_causal_raises_unoverridden_lorentzian(self):
        class NoCausal(sp.Spacetime):
            def dimension(self):
                return 2

            def volume(self):
                return 1.0

            def sample(self, rng, n):
                return rng.uniform(0.0, 1.0, size=(n, 2))

        with self.assertRaises(NotImplementedError):
            NoCausal().is_causal([0.0, 0.0], [1.0, 1.0])

    def test_scalar_coeffs_raises_by_default(self):
        with self.assertRaises(NotImplementedError):
            _Lorentzian().scalar_coeffs(1.0, 10.0)

    def test_to_embedding_default_identity(self):
        coords = [[1.0, 2.0, 3.0]]
        self.assertEqual(sp.Spacetime.to_embedding(_Lorentzian(), coords), coords)

    def test_boundary_default_empty(self):
        self.assertEqual(_Lorentzian().boundary(), [])

    def test_bad_signature_shape_rejected(self):
        with self.assertRaises(TypeError):

            class BadSig(sp.Spacetime):  # noqa: F841
                signature = (1, 2, 3)

                def dimension(self):
                    return 3

                def volume(self):
                    return 1.0

                def sample(self, rng, n):
                    return rng.uniform(0.0, 1.0, size=(n, 3))

    def test_negative_signature_rejected(self):
        with self.assertRaises(ValueError):

            class BadSign(sp.Spacetime):  # noqa: F841
                signature = (-1, 3)

                def dimension(self):
                    return 2

                def volume(self):
                    return 1.0

                def sample(self, rng, n):
                    return rng.uniform(0.0, 1.0, size=(n, 2))


class TestRegistry(unittest.TestCase):
    def test_register_and_lookup(self):
        class A(sp.Spacetime):
            def dimension(self):
                return 2

            def volume(self):
                return 1.0

            def sample(self, rng, n):
                return rng.uniform(0.0, 1.0, size=(n, 2))

        sp.register("r2_test_register")(A)
        self.assertIs(sp.get_registry()["r2_test_register"], A)

    def test_collision_raises(self):
        class A(sp.Spacetime):
            def dimension(self):
                return 2

            def volume(self):
                return 1.0

            def sample(self, rng, n):
                return rng.uniform(0.0, 1.0, size=(n, 2))

        sp.register("r2_test_collision")(A)
        with self.assertRaises(ValueError):
            sp.register("r2_test_collision")(_Lorentzian)

    def test_overwrite_replaces(self):
        class A(sp.Spacetime):
            def dimension(self):
                return 2

            def volume(self):
                return 1.0

            def sample(self, rng, n):
                return rng.uniform(0.0, 1.0, size=(n, 2))

        class B(sp.Spacetime):
            def dimension(self):
                return 2

            def volume(self):
                return 1.0

            def sample(self, rng, n):
                return rng.uniform(0.0, 1.0, size=(n, 2))

        sp.register("r2_test_overwrite")(A)
        sp.register("r2_test_overwrite", overwrite=True)(B)
        self.assertIs(sp.get_registry()["r2_test_overwrite"], B)

    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            sp.register("")(_Lorentzian)
        with self.assertRaises(ValueError):
            sp.register(123)(_Lorentzian)


class TestCreate(unittest.TestCase):
    def test_dimension_too_small(self):
        with self.assertRaises(ValueError):
            sp.create(dimension=1, domain="diamond")

    def test_signature_mismatch(self):
        with self.assertRaises(ValueError):
            sp.create(dimension=3, signature=(2, 2), domain="diamond")

    def test_unsupported_metric(self):
        with self.assertRaises(NotImplementedError):
            sp.create(dimension=2, domain="diamond", metric="de_sitter")

    def test_unsupported_domain(self):
        with self.assertRaises(NotImplementedError):
            sp.create(dimension=2, domain="ball")

    def test_missing_cylinder_params(self):
        with self.assertRaises(ValueError):
            sp.create(dimension=2, domain="cylinder")

    def test_missing_box_params(self):
        with self.assertRaises(ValueError):
            sp.create(dimension=2, domain="box")

    def test_create_diamond_native(self):
        st = sp.create(dimension=2, domain="diamond")
        self.assertEqual(st.dimension(), 2)
        self.assertEqual(st.signature, (1, 1))

    def test_native_classes_have_signature(self):
        self.assertEqual(sp.MinkowskiDiamond(2).signature, (1, 1))
        self.assertEqual(sp.MinkowskiCylinder(2, 2.0, 3.0).signature, (1, 1))
        self.assertEqual(sp.MinkowskiBox(2, 2.0, 1.0).signature, (1, 1))


if __name__ == "__main__":
    unittest.main()
