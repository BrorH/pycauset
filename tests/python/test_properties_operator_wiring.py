import unittest
import pycauset


class TestPropertiesOperatorWiring(unittest.TestCase):
    def test_solve_zero_property_is_singular(self):
        a = pycauset.DenseMatrix(2, dtype="float64")
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
