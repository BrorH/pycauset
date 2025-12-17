import os
import shutil
import tempfile
import unittest

import numpy as np

import pycauset
from pycauset import save


class TestComplexVectorDTypes(unittest.TestCase):
    def test_construct_set_get_complex_float32(self):
        v = pycauset.empty(3, dtype="complex_float32")
        self.assertTrue("ComplexFloat32Vector" in str(v))

        v[0] = 1.25 + 2.5j
        v[1] = -3.0 + 0.5j

        z0 = v[0]
        z1 = v[1]
        self.assertAlmostEqual(z0.real, 1.25, places=5)
        self.assertAlmostEqual(z0.imag, 2.5, places=5)
        self.assertAlmostEqual(z1.real, -3.0, places=5)
        self.assertAlmostEqual(z1.imag, 0.5, places=5)

        v.close()

    def test_numpy_roundtrip_complex64(self):
        arr = np.array([1 + 2j, 3 - 4j, 0.25 + 0.5j], dtype=np.complex64)
        v = pycauset.vector(arr)
        self.assertTrue("ComplexFloat32Vector" in str(v))

        out = np.array(v)
        self.assertEqual(out.dtype, np.complex64)
        self.assertTrue(np.allclose(out, arr))

        v.close()

    def test_ops_match_numpy_complex64(self):
        a = np.array([1 + 2j, 3 - 4j, -0.25 + 0.5j], dtype=np.complex64)
        b = np.array([-2 + 1j, 0.5 + 0.25j, 4 + 0j], dtype=np.complex64)

        va = pycauset.vector(a)
        vb = pycauset.vector(b)

        vc = va + vb
        vd = va - vb

        self.assertTrue(np.allclose(np.array(vc), a + b))
        self.assertTrue(np.allclose(np.array(vd), a - b))

        va.close()
        vb.close()
        vc.close()
        vd.close()

    def test_scalar_mul_complex64(self):
        a = np.array([1 + 2j, 3 - 4j, -0.25 + 0.5j], dtype=np.complex64)
        s = 1.5 - 0.5j

        va = pycauset.vector(a)
        vb = va * s

        self.assertTrue(np.allclose(np.array(vb), a * s))

        va.close()
        vb.close()

    def test_complex_float16_numpy_view_and_persistence(self):
        tmp_dir = tempfile.mkdtemp(prefix="pycauset_cf16_vec_")
        try:
            v = pycauset.vector([1 + 2j, 3 - 4j, 0.25 + 0.5j], dtype="complex_float16")
            self.assertTrue("ComplexFloat16Vector" in str(v))

            out = np.array(v)
            self.assertEqual(out.dtype, np.complex64)
            expected = np.array([1 + 2j, 3 - 4j, 0.25 + 0.5j], dtype=np.complex64)
            self.assertTrue(np.allclose(out, expected, atol=5e-2, rtol=5e-2))

            path = os.path.join(tmp_dir, "v.pycauset")
            save(v, path)
            v.close()

            v2 = pycauset.load(path)
            self.assertTrue("ComplexFloat16Vector" in str(v2))
            out2 = np.array(v2)
            self.assertEqual(out2.dtype, np.complex64)
            self.assertTrue(np.allclose(out2, expected, atol=5e-2, rtol=5e-2))
            v2.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_dot_complex64_matches_numpy(self):
        a = np.array([1 + 2j, 3 - 4j, -0.25 + 0.5j], dtype=np.complex64)
        b = np.array([-2 + 1j, 0.5 + 0.25j, 4 + 0j], dtype=np.complex64)

        va = pycauset.vector(a)
        vb = pycauset.vector(b)

        got_matmul = va @ vb
        got_method = va.dot(vb)

        expected = np.sum(a * b)
        self.assertAlmostEqual(got_matmul.real, expected.real, places=5)
        self.assertAlmostEqual(got_matmul.imag, expected.imag, places=5)
        self.assertAlmostEqual(got_method.real, expected.real, places=5)
        self.assertAlmostEqual(got_method.imag, expected.imag, places=5)

        va.close()
        vb.close()

    def test_matvec_vecmat_outer_complex64_matches_numpy(self):
        m = np.array(
            [
                [1 + 0.5j, 2 - 1j, 0.25 + 0j],
                [-3 + 0j, 0 + 2j, 1 - 0.25j],
                [0.5 + 1j, -1 + 0j, 2 + 0.75j],
            ],
            dtype=np.complex64,
        )
        v = np.array([1 + 2j, 3 - 4j, -0.25 + 0.5j], dtype=np.complex64)

        pm = pycauset.matrix(m)
        pv = pycauset.vector(v)

        got_mv = pm @ pv
        got_vm = pv @ pm
        got_outer = pv @ pv.T

        self.assertTrue(np.allclose(np.array(got_mv), m @ v))
        self.assertTrue(np.allclose(np.array(got_vm), v @ m))
        self.assertTrue(np.allclose(np.array(got_outer), np.outer(v, v)))

        pm.close()
        pv.close()
        got_mv.close()
        got_vm.close()
        got_outer.close()

    def test_matvec_complex_float16_tolerant(self):
        m = np.array(
            [
                [1 + 0.5j, 2 - 1j, 0.25 + 0j],
                [-3 + 0j, 0 + 2j, 1 - 0.25j],
                [0.5 + 1j, -1 + 0j, 2 + 0.75j],
            ],
            dtype=np.complex64,
        )
        v = np.array([1 + 2j, 3 - 4j, -0.25 + 0.5j], dtype=np.complex64)

        pm = pycauset.matrix(m, dtype="complex_float16")
        pv = pycauset.vector(v, dtype="complex_float16")

        got = pm @ pv
        expected = (m @ v).astype(np.complex64)
        self.assertTrue(np.allclose(np.array(got), expected, atol=5e-2, rtol=5e-2))

        pm.close()
        pv.close()
        got.close()
