import unittest

import pycauset


class TestPropertiesCompatibility(unittest.TestCase):
    def test_unitary_requires_square_matrix(self):
        m = None
        try:
            m = pycauset.FloatMatrix(2, 3)
            with self.assertRaises(ValueError):
                m.properties["is_unitary"] = True
        finally:
            if m is not None:
                m.close()

    def test_cached_trace_requires_square_matrix(self):
        m = None
        try:
            m = pycauset.FloatMatrix(2, 3)
            with self.assertRaises(ValueError):
                m.properties["trace"] = 1.0
        finally:
            if m is not None:
                m.close()

    def test_identity_contradicts_zero(self):
        m = None
        try:
            m = pycauset.FloatMatrix(2)
            m.properties["is_identity"] = True
            with self.assertRaises(ValueError):
                m.properties["is_zero"] = True
        finally:
            if m is not None:
                m.close()

    def test_diagonal_implies_not_upper_false(self):
        m = None
        try:
            m = pycauset.FloatMatrix(2)
            m.properties["is_diagonal"] = True
            with self.assertRaises(ValueError):
                m.properties["is_upper_triangular"] = False
        finally:
            if m is not None:
                m.close()

    def test_identity_implies_not_permutation_false(self):
        m = None
        try:
            m = pycauset.FloatMatrix(2)
            m.properties["is_identity"] = True
            with self.assertRaises(ValueError):
                m.properties["is_permutation"] = False
        finally:
            if m is not None:
                m.close()

    def test_property_setter_validates(self):
        m = None
        try:
            m = pycauset.FloatMatrix(2, 3)
            with self.assertRaises(ValueError):
                m.properties = {"is_unitary": True}
        finally:
            if m is not None:
                m.close()

    def test_upper_triangular_requires_square_matrix(self):
        m = None
        try:
            m = pycauset.FloatMatrix(2, 3)
            with self.assertRaises(ValueError):
                m.properties["is_upper_triangular"] = True
        finally:
            if m is not None:
                m.close()

    def test_atomic_requires_square_matrix(self):
        m = None
        try:
            m = pycauset.FloatMatrix(2, 3)
            with self.assertRaises(ValueError):
                m.properties["is_atomic"] = True
        finally:
            if m is not None:
                m.close()

    def test_strictly_sorted_implies_sorted(self):
        v = None
        try:
            v = pycauset.vector([1.0, 2.0, 3.0], dtype="float64")
            with self.assertRaises(ValueError):
                v.properties = {"is_strictly_sorted": True, "is_sorted": False}
        finally:
            if v is not None:
                v.close()

    def test_upper_and_lower_true_contradicts_diagonal_false(self):
        m = None
        try:
            m = pycauset.FloatMatrix(3)
            with self.assertRaises(ValueError):
                m.properties = {"is_upper_triangular": True, "is_lower_triangular": True, "is_diagonal": False}
        finally:
            if m is not None:
                m.close()


class TestCachedDerivedPropagation(unittest.TestCase):
    def test_scalar_multiply_propagates_cached_trace(self):
        m = None
        m2 = None
        try:
            m = pycauset.FloatMatrix(3)
            m.properties["trace"] = 7.0
            m2 = 2.0 * m
            self.assertEqual(m2.properties.get("trace"), 14.0)
        finally:
            if m2 is not None:
                m2.close()
            if m is not None:
                m.close()

    def test_scalar_multiply_propagates_cached_determinant(self):
        m = None
        m2 = None
        try:
            m = pycauset.FloatMatrix(3)
            m.properties["determinant"] = 3.0
            m2 = 2.0 * m
            self.assertEqual(m2.properties.get("determinant"), 3.0 * (2.0**3))
        finally:
            if m2 is not None:
                m2.close()
            if m is not None:
                m.close()

    def test_scalar_multiply_clears_identity(self):
        m = None
        m2 = None
        try:
            m = pycauset.FloatMatrix(3)
            m.properties["is_identity"] = True
            m2 = 2.0 * m
            self.assertIn("is_identity", m2.properties)
            self.assertIs(m2.properties.get("is_identity"), False)
        finally:
            if m2 is not None:
                m2.close()
            if m is not None:
                m.close()

    def test_add_propagates_trace_when_both_known(self):
        a = None
        b = None
        c = None
        try:
            a = pycauset.FloatMatrix(2)
            b = pycauset.FloatMatrix(2)
            a.properties["trace"] = 1.0
            b.properties["trace"] = 2.0
            c = a + b
            self.assertEqual(c.properties.get("trace"), 3.0)
            self.assertNotIn("determinant", c.properties)
        finally:
            if c is not None:
                c.close()
            if b is not None:
                b.close()
            if a is not None:
                a.close()

    def test_add_propagates_sum_when_both_known(self):
        a = None
        b = None
        c = None
        try:
            a = pycauset.ComplexFloat64Matrix(2)
            b = pycauset.ComplexFloat64Matrix(2)
            a.properties["sum"] = 1 + 2j
            b.properties["sum"] = 3 + 4j
            c = a + b
            self.assertEqual(c.properties.get("sum"), (1 + 2j) + (3 + 4j))
        finally:
            if c is not None:
                c.close()
            if b is not None:
                b.close()
            if a is not None:
                a.close()

    def test_scalar_zero_propagates_cached_values_when_present(self):
        m = None
        m2 = None
        try:
            m = pycauset.FloatMatrix(3)
            m.properties["trace"] = 7.0
            m.properties["determinant"] = 3.0
            m.properties["norm"] = 5.0
            m.properties["sum"] = 9.0
            m.properties["rank"] = 2

            m2 = 0.0 * m

            self.assertEqual(m2.properties.get("trace"), 0.0)
            self.assertEqual(m2.properties.get("determinant"), 0.0)
            self.assertEqual(m2.properties.get("norm"), 0.0)
            self.assertEqual(m2.properties.get("sum"), 0.0)
            self.assertEqual(m2.properties.get("rank"), 0)

            # Do not infer new gospel True claims from scalar=0.
            self.assertNotIn("is_zero", m2.properties)
        finally:
            if m2 is not None:
                m2.close()
            if m is not None:
                m.close()


if __name__ == "__main__":
    unittest.main()
