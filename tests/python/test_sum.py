import unittest
import tempfile

import numpy as np

import pycauset


class TestSum(unittest.TestCase):
    def test_sum_vector_float(self):
        v = pycauset.FloatVector(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        try:
            self.assertEqual(pycauset.sum(v), 6.0 + 0.0j)

            props = getattr(v, "properties", None)
            self.assertIsNotNone(props)
            self.assertEqual(props.get("sum"), 6.0 + 0.0j)
        finally:
            v.close()

    def test_sum_matrix_float(self):
        m = pycauset.FloatMatrix(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        try:
            self.assertEqual(pycauset.sum(m), 10.0 + 0.0j)

            props = getattr(m, "properties", None)
            self.assertIsNotNone(props)
            self.assertEqual(props.get("sum"), 10.0 + 0.0j)
        finally:
            m.close()

    def test_sum_scalar_multiply_propagates_cached_sum(self):
        v = pycauset.FloatVector(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        v2 = None
        try:
            self.assertEqual(pycauset.sum(v), 6.0 + 0.0j)
            v2 = v * 2.0
            props2 = getattr(v2, "properties", None)
            self.assertIsNotNone(props2)
            self.assertEqual(props2.get("sum"), 12.0 + 0.0j)

            # Confirm the public API can serve from the propagated cache.
            self.assertEqual(pycauset.sum(v2), 12.0 + 0.0j)
        finally:
            if v2 is not None:
                v2.close()
            v.close()

    def test_sum_persists_as_cached_derived(self):
        m = pycauset.FloatMatrix(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        try:
            self.assertEqual(pycauset.sum(m), 10.0 + 0.0j)
            props = getattr(m, "properties", None)
            self.assertIsNotNone(props)
            self.assertEqual(props.get("sum"), 10.0 + 0.0j)

            with tempfile.TemporaryDirectory(prefix="pycauset_test_sum_") as tmp_dir:
                path = f"{tmp_dir}/m.pycauset"
                pycauset.save(m, path)
                m2 = pycauset.load(path)
                try:
                    props2 = getattr(m2, "properties", None)
                    self.assertIsNotNone(props2)
                    self.assertEqual(props2.get("sum"), 10.0 + 0.0j)
                finally:
                    # Windows can't delete open mmap-backed files.
                    m2.close()
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
