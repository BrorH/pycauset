import unittest
import pycauset


class TestPropertiesOperatorWiring(unittest.TestCase):
    def test_solve_zero_property_is_singular(self):
        a = pycauset.DenseMatrix(2, dtype="float64")
        b = pycauset.vector([1.0, 1.0])
        try:
            # Properties never change results (F-090): the zero matrix is singular because of
            # its values, and the error is the engine's, with or without the property.
            a.properties = {"is_zero": True}
            with self.assertRaises(RuntimeError):
                pycauset.solve(a, b)
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
