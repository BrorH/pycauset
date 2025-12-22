import unittest
import pycauset


class TestPropertiesOperatorWiring(unittest.TestCase):
    def test_trace_prefers_properties_even_if_wrong(self):
        m = pycauset.FloatMatrix(2)
        try:
            m.set(0, 0, 1.0)
            m.set(1, 1, 2.0)

            m.properties = {"trace": 999.0}
            self.assertEqual(m.trace(), 999.0)
        finally:
            m.close()

    def test_determinant_prefers_properties_even_if_wrong(self):
        m = pycauset.FloatMatrix(2)
        try:
            m.set(0, 0, 1.0)
            m.set(1, 1, 2.0)

            m.properties = {"determinant": -123.0}
            self.assertEqual(m.determinant(), -123.0)
        finally:
            m.close()

    def test_identity_shortcuts_are_gospel(self):
        m = pycauset.FloatMatrix(3)
        try:
            m.set(0, 0, 2.0)
            m.set(1, 1, 3.0)
            m.set(2, 2, 4.0)

            m.properties = {"is_identity": True}
            self.assertEqual(m.trace(), 3)
            self.assertEqual(m.determinant(), 1)
        finally:
            m.close()

    def test_diagonal_value_shortcuts(self):
        m = pycauset.FloatMatrix(4)
        try:
            m.properties = {"is_diagonal": True, "diagonal_value": 3.0}
            self.assertEqual(m.trace(), 12.0)
            self.assertEqual(m.determinant(), 81.0)

            # Derived values get populated into properties for persistence.
            self.assertEqual(m.properties.get("trace"), 12.0)
            self.assertEqual(m.properties.get("determinant"), 81.0)
        finally:
            m.close()

    def test_solve_identity_property_shortcut(self):
        a = pycauset.FloatMatrix(2)
        b = pycauset.vector([5.0, -2.0])
        try:
            # Payload is not identity, but properties are gospel.
            a.set(0, 0, 10.0)
            a.set(0, 1, 7.0)
            a.set(1, 0, 3.0)
            a.set(1, 1, 4.0)
            a.properties = {"is_identity": True}

            x = pycauset.solve(a, b)
            try:
                self.assertAlmostEqual(x.get(0), 5.0)
                self.assertAlmostEqual(x.get(1), -2.0)
            finally:
                x.close()
        finally:
            a.close()
            b.close()

    def test_solve_zero_property_is_singular(self):
        a = pycauset.FloatMatrix(2)
        b = pycauset.vector([1.0, 1.0])
        try:
            a.properties = {"is_zero": True}
            with self.assertRaises(ValueError):
                pycauset.solve(a, b)
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
