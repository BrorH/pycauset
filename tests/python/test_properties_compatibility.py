import unittest

import pycauset


class TestPropertiesCompatibility(unittest.TestCase):
    def test_unitary_requires_square_matrix(self):
        m = None
        try:
            m = pycauset.DenseMatrix(2, 3, dtype="float64")
            with self.assertRaises(ValueError):
                m.properties["is_unitary"] = True
        finally:
            if m is not None:
                m.close()

    def test_identity_contradicts_zero(self):
        m = None
        try:
            m = pycauset.DenseMatrix(2, dtype="float64")
            m.properties["is_identity"] = True
            with self.assertRaises(ValueError):
                m.properties["is_zero"] = True
        finally:
            if m is not None:
                m.close()

    def test_diagonal_implies_not_upper_false(self):
        m = None
        try:
            m = pycauset.DenseMatrix(2, dtype="float64")
            m.properties["is_diagonal"] = True
            with self.assertRaises(ValueError):
                m.properties["is_upper_triangular"] = False
        finally:
            if m is not None:
                m.close()

    def test_identity_implies_not_permutation_false(self):
        m = None
        try:
            m = pycauset.DenseMatrix(2, dtype="float64")
            m.properties["is_identity"] = True
            with self.assertRaises(ValueError):
                m.properties["is_permutation"] = False
        finally:
            if m is not None:
                m.close()

    def test_property_setter_validates(self):
        m = None
        try:
            m = pycauset.DenseMatrix(2, 3, dtype="float64")
            with self.assertRaises(ValueError):
                m.properties = {"is_unitary": True}
        finally:
            if m is not None:
                m.close()

    def test_upper_triangular_requires_square_matrix(self):
        m = None
        try:
            m = pycauset.DenseMatrix(2, 3, dtype="float64")
            with self.assertRaises(ValueError):
                m.properties["is_upper_triangular"] = True
        finally:
            if m is not None:
                m.close()

    def test_atomic_requires_square_matrix(self):
        m = None
        try:
            m = pycauset.DenseMatrix(2, 3, dtype="float64")
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
            m = pycauset.DenseMatrix(3, dtype="float64")
            with self.assertRaises(ValueError):
                m.properties = {"is_upper_triangular": True, "is_lower_triangular": True, "is_diagonal": False}
        finally:
            if m is not None:
                m.close()


if __name__ == "__main__":
    unittest.main()
