import tempfile
import unittest

import pycauset


class TestNorm(unittest.TestCase):
    def test_norm_vector_l2(self):
        v = pycauset.vector([3.0, 4.0], dtype="float64")
        assert pycauset.norm(v) == 5.0

    def test_norm_vector_scaling_respects_scalar(self):
        v = pycauset.vector([3.0, 4.0], dtype="float64")
        v2 = v * 2.0
        assert pycauset.norm(v2) == 10.0

    def test_norm_matrix_frobenius(self):
        m = pycauset.matrix([[3.0, 4.0], [0.0, 0.0]], dtype="float64")
        assert pycauset.norm(m) == 5.0

    def test_norm_caches_value_in_properties(self):
        v = pycauset.vector([3.0, 4.0], dtype="float64")

        props = getattr(v, "properties", None)
        self.assertIsNotNone(props)

        assert props.get("norm") is None
        assert pycauset.norm(v) == 5.0
        assert props.get("norm") == 5.0

        # Second call should be consistent (and may hit the cache).
        assert pycauset.norm(v) == 5.0

    def test_norm_persists_as_cached_derived(self):
        m = pycauset.matrix([[3.0, 4.0], [0.0, 0.0]], dtype="float64")
        assert pycauset.norm(m) == 5.0

        props = getattr(m, "properties", None)
        self.assertIsNotNone(props)
        assert props.get("norm") == 5.0

        with tempfile.TemporaryDirectory(prefix="pycauset_test_norm_") as tmp_dir:
            path = f"{tmp_dir}/m.pycauset"
            pycauset.save(m, path)
            m2 = pycauset.load(path)
            try:
                props2 = getattr(m2, "properties", None)
                self.assertIsNotNone(props2)
                assert props2.get("norm") == 5.0
            finally:
                try:
                    m2.close()
                except Exception:
                    pass


def test_norm_vector_l2():
    v = pycauset.vector([3.0, 4.0], dtype="float64")
    assert pycauset.norm(v) == 5.0


def test_norm_vector_scaling_respects_scalar():
    v = pycauset.vector([3.0, 4.0], dtype="float64")
    v2 = v * 2.0
    assert pycauset.norm(v2) == 10.0


def test_norm_matrix_frobenius():
    m = pycauset.matrix([[3.0, 4.0], [0.0, 0.0]], dtype="float64")
    assert pycauset.norm(m) == 5.0
