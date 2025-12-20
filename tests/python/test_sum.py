import unittest

import numpy as np

import pycauset


class TestSum(unittest.TestCase):
    def test_sum_vector_float(self):
        v = pycauset.FloatVector(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        try:
            self.assertEqual(pycauset.sum(v), 6.0 + 0.0j)
        finally:
            v.close()

    def test_sum_matrix_float(self):
        m = pycauset.FloatMatrix(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        try:
            self.assertEqual(pycauset.sum(m), 10.0 + 0.0j)
        finally:
            m.close()

    def test_sum_vector_complex_and_conjugation(self):
        if getattr(pycauset, "ComplexFloat64Vector", None) is None:
            self.skipTest("ComplexFloat64Vector is not available")

        arr = np.array([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex128)
        v = pycauset.ComplexFloat64Vector(arr)
        vc = None
        try:
            expected = (1 + 2j) + (3 - 4j) + (-5 + 6j)
            self.assertEqual(pycauset.sum(v), expected)

            vc = v.conj()
            self.assertEqual(pycauset.sum(vc), np.conj(expected))
        finally:
            if vc is not None:
                vc.close()
            v.close()


if __name__ == "__main__":
    unittest.main()
