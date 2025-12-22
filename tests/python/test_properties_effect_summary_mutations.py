import unittest

import pycauset


class TestPropertiesEffectSummaryMutations(unittest.TestCase):
    def test_set_off_diagonal_clears_diagonal_claim(self):
        m = pycauset.FloatMatrix(3)
        try:
            m.properties = {"is_diagonal": True}
            m.set(0, 1, 1.0)
            self.assertIs(m.properties.get("is_diagonal"), False)
        finally:
            m.close()

    def test_set_below_diagonal_clears_upper_triangular_claim(self):
        m = pycauset.FloatMatrix(3)
        try:
            m.properties = {"is_upper_triangular": True}
            m.set(2, 0, 1.0)  # below diagonal
            self.assertIs(m.properties.get("is_upper_triangular"), False)
        finally:
            m.close()

    def test_set_identity_sets_identity_properties(self):
        m = pycauset.FloatMatrix(3)
        try:
            m.properties = {}
            m.set_identity()
            self.assertIs(m.properties.get("is_identity"), True)
            self.assertIs(m.properties.get("is_diagonal"), True)
            self.assertEqual(m.properties.get("diagonal_value"), 1)
        finally:
            m.close()


if __name__ == "__main__":
    unittest.main()
