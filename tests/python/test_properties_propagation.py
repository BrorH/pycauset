import unittest
import pycauset


class TestPropertiesPropagation(unittest.TestCase):
    def test_transpose_propagates_properties(self):
        m = pycauset.ComplexFloat64Matrix(2)
        mt = None
        try:
            m.properties = {
                "is_upper_triangular": True,
                "is_lower_triangular": False,
                "is_unitary": True,
                "is_hermitian": True,
                "diagonal_value": 1 + 2j,
                "trace": 3 + 4j,
            }

            mt = m.T

            # Independent ownership
            self.assertIsNot(m.properties, mt.properties)

            # Structural swap
            self.assertEqual(mt.properties.get("is_lower_triangular"), True)
            self.assertEqual(mt.properties.get("is_upper_triangular"), False)

            # Unitarity cleared on transpose (becomes None/unset)
            self.assertNotIn("is_unitary", mt.properties)

            # Hermitian cleared for complex on transpose
            self.assertNotIn("is_hermitian", mt.properties)

            # diagonal_value preserved
            self.assertEqual(mt.properties.get("diagonal_value"), 1 + 2j)

            # trace preserved
            self.assertEqual(mt.properties.get("trace"), 3 + 4j)

            # parent unchanged
            self.assertIn("is_unitary", m.properties)
            self.assertIn("is_hermitian", m.properties)
        finally:
            if mt is not None:
                mt.close()
            m.close()

    def test_conj_propagates_properties(self):
        m = pycauset.ComplexFloat64Matrix(2)
        mc = None
        try:
            m.properties = {
                "is_unitary": True,
                "is_hermitian": True,
                "diagonal_value": 1 + 2j,
                "trace": 3 + 4j,
                "determinant": 5 + 6j,
                "sum": 7 + 8j,
            }

            mc = m.conj()

            # diagonal_value conjugated
            self.assertEqual(mc.properties.get("diagonal_value"), 1 - 2j)

            # derived values conjugated
            self.assertEqual(mc.properties.get("trace"), 3 - 4j)
            self.assertEqual(mc.properties.get("determinant"), 5 - 6j)
            self.assertEqual(mc.properties.get("sum"), 7 - 8j)

            # Unitarity cleared on conj
            self.assertNotIn("is_unitary", mc.properties)

            # Hermitian cleared for complex on conj
            self.assertNotIn("is_hermitian", mc.properties)
        finally:
            if mc is not None:
                mc.close()
            m.close()

    def test_adjoint_propagates_properties_with_exceptions(self):
        m = pycauset.ComplexFloat64Matrix(2)
        mh = None
        try:
            m.properties = {
                "is_upper_triangular": True,
                "is_lower_triangular": False,
                "is_unitary": True,
                "is_hermitian": True,
                "diagonal_value": 1 + 2j,
                "trace": 3 + 4j,
                "sum": 7 + 8j,
            }

            mh = m.H

            # Swap due to transpose component
            self.assertEqual(mh.properties.get("is_lower_triangular"), True)
            self.assertEqual(mh.properties.get("is_upper_triangular"), False)

            # diagonal_value conjugated
            self.assertEqual(mh.properties.get("diagonal_value"), 1 - 2j)

            # trace conjugated
            self.assertEqual(mh.properties.get("trace"), 3 - 4j)
            self.assertEqual(mh.properties.get("sum"), 7 - 8j)

            # Exceptions: these stay the same for adjoint
            self.assertEqual(mh.properties.get("is_unitary"), True)
            self.assertEqual(mh.properties.get("is_hermitian"), True)
        finally:
            if mh is not None:
                mh.close()
            m.close()


if __name__ == "__main__":
    unittest.main()
