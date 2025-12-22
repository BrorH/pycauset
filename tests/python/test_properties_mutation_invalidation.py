import unittest
import pycauset


class TestPropertiesMutationInvalidation(unittest.TestCase):
    def test_trace_cache_is_cleared_on_set(self):
        m = pycauset.FloatMatrix(2)
        try:
            m.set(0, 0, 1.0)
            m.set(1, 1, 2.0)

            self.assertEqual(m.trace(), 3.0)
            self.assertEqual(m.properties.get("trace"), 3.0)

            m.set(0, 0, 100.0)
            self.assertNotIn("trace", m.properties)
            self.assertEqual(m.trace(), 102.0)
        finally:
            m.close()

    def test_determinant_cache_is_cleared_on_set(self):
        m = pycauset.FloatMatrix(2)
        try:
            m.set(0, 0, 1.0)
            m.set(1, 1, 2.0)

            self.assertEqual(m.determinant(), 2.0)
            self.assertEqual(m.properties.get("determinant"), 2.0)

            m.set(0, 0, 3.0)
            self.assertNotIn("determinant", m.properties)
            self.assertEqual(m.determinant(), 6.0)
        finally:
            m.close()

    def test_inverse_cache_attribute_is_cleared_on_set_best_effort(self):
        m = pycauset.FloatMatrix(2)
        try:
            try:
                m._cached_inverse = object()
            except Exception:
                self.skipTest("native object does not accept attributes")

            m.set(0, 0, 1.0)
            self.assertFalse(hasattr(m, "_cached_inverse"))
        finally:
            m.close()


if __name__ == "__main__":
    unittest.main()
