import unittest

import pycauset
from pycauset._internal.properties import effective_structure


class TestEffectiveStructure(unittest.TestCase):
    def test_priority_tree_zero_identity_diagonal(self):
        m = None
        try:
            m = pycauset.FloatMatrix(3)

            m.properties = {"is_zero": True}
            self.assertEqual(effective_structure(m), "zero")

            m.properties = {"is_identity": True}
            self.assertEqual(effective_structure(m), "identity")

            m.properties = {"is_diagonal": True}
            self.assertEqual(effective_structure(m), "diagonal")

            m.properties = {"is_upper_triangular": True, "is_lower_triangular": True}
            self.assertEqual(effective_structure(m), "diagonal")
        finally:
            if m is not None:
                m.close()

    def test_priority_tree_triangular_and_general(self):
        m = None
        try:
            m = pycauset.FloatMatrix(3)

            m.properties = {"is_upper_triangular": True}
            self.assertEqual(effective_structure(m), "upper_triangular")

            m.properties = {"is_lower_triangular": True}
            self.assertEqual(effective_structure(m), "lower_triangular")

            m.properties = {}
            self.assertEqual(effective_structure(m), "general")
        finally:
            if m is not None:
                m.close()


if __name__ == "__main__":
    unittest.main()
