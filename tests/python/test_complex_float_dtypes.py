import unittest
import sys
from pathlib import Path
import tempfile

# Allow running tests from a source checkout without installing.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    _p = str(_path)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import pycauset


class TestComplexFloatDTypes(unittest.TestCase):
    def test_construct_set_get_complex_float32(self):
        if getattr(pycauset, "ComplexFloat32Matrix", None) is None:
            self.skipTest("ComplexFloat32Matrix is not available")

        a = pycauset.Matrix(2, dtype=pycauset.complex_float32)
        self.assertIsInstance(a, pycauset.ComplexFloat32Matrix)

        a[0, 0] = 1 + 2j
        a[1, 1] = 3 - 4j
        self.assertEqual(a[0, 0], 1 + 2j)
        self.assertEqual(a[1, 1], 3 - 4j)

    def test_numpy_roundtrip_complex64(self):
        if getattr(pycauset, "ComplexFloat32Matrix", None) is None:
            self.skipTest("ComplexFloat32Matrix is not available")

        arr = np.array([[1 + 2j, 0 + 0j], [0 + 0j, 3 - 4j]], dtype=np.complex64)
        m = pycauset.Matrix(arr)
        self.assertIsInstance(m, pycauset.ComplexFloat32Matrix)

        out = np.array(m)
        self.assertEqual(out.dtype, np.dtype(np.complex64))
        self.assertTrue(np.allclose(out, arr))

    def test_numpy_roundtrip_complex128(self):
        if getattr(pycauset, "ComplexFloat64Matrix", None) is None:
            self.skipTest("ComplexFloat64Matrix is not available")

        arr = np.array([[1 + 2j, 0 + 0j], [0 + 0j, 3 - 4j]], dtype=np.complex128)
        m = pycauset.Matrix(arr)
        self.assertIsInstance(m, pycauset.ComplexFloat64Matrix)

        out = np.array(m)
        self.assertEqual(out.dtype, np.dtype(np.complex128))
        self.assertTrue(np.allclose(out, arr))

    def test_ops_match_numpy_complex64(self):
        if getattr(pycauset, "ComplexFloat32Matrix", None) is None:
            self.skipTest("ComplexFloat32Matrix is not available")

        a0 = np.array([[1 + 2j, 2 - 1j], [0.5 + 0j, -3 + 4j]], dtype=np.complex64)
        b0 = np.array([[5 + 0j, 0 + 6j], [7 - 2j, 1 + 0j]], dtype=np.complex64)

        a = pycauset.Matrix(a0)
        b = pycauset.Matrix(b0)

        add_out = np.array(a + b)
        mul_out = np.array(a * b)
        mm_out = np.array(a @ b)

        self.assertTrue(np.allclose(add_out, a0 + b0))
        self.assertTrue(np.allclose(mul_out, a0 * b0))
        self.assertTrue(np.allclose(mm_out, a0 @ b0))

    def test_complex_float16_array_protocol(self):
        if getattr(pycauset, "ComplexFloat16Matrix", None) is None:
            self.skipTest("ComplexFloat16Matrix is not available")

        m = pycauset.Matrix(2, dtype=pycauset.complex_float16)
        self.assertIsInstance(m, pycauset.ComplexFloat16Matrix)

        m[0, 0] = 1.5 + 2.25j
        m[1, 1] = -3.0 + 4.0j

        z00 = m[0, 0]
        z11 = m[1, 1]
        self.assertAlmostEqual(z00.real, 1.5, places=2)
        self.assertAlmostEqual(z00.imag, 2.25, places=2)
        self.assertAlmostEqual(z11.real, -3.0, places=2)
        self.assertAlmostEqual(z11.imag, 4.0, places=2)

        out = np.array(m)
        # complex_float16 is exposed as complex64 in NumPy interop.
        self.assertEqual(out.dtype, np.dtype(np.complex64))
        self.assertTrue(np.allclose(out[0, 0], np.complex64(1.5 + 2.25j), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.allclose(out[1, 1], np.complex64(-3.0 + 4.0j), atol=1e-2, rtol=1e-2))

    def test_ops_match_numpy_complex_float16(self):
        if getattr(pycauset, "ComplexFloat16Matrix", None) is None:
            self.skipTest("ComplexFloat16Matrix is not available")

        a0 = np.array(
            [[1.25 + 2.5j, 2.0 - 1.0j], [0.5 + 0.0j, -3.0 + 4.0j]],
            dtype=np.complex64,
        )
        b0 = np.array(
            [[5.0 + 0.0j, 0.0 + 6.0j], [7.0 - 2.0j, 1.0 + 0.0j]],
            dtype=np.complex64,
        )

        a = pycauset.Matrix(2, dtype=pycauset.complex_float16)
        b = pycauset.Matrix(2, dtype=pycauset.complex_float16)

        for i in range(2):
            for j in range(2):
                a[i, j] = a0[i, j]
                b[i, j] = b0[i, j]

        add_out = np.array(a + b)
        mul_out = np.array(a * b)
        mm_out = np.array(a @ b)
        sm_out = np.array((1.5 - 0.25j) * a)

        # complex_float16 is stored in half precision, so allow modest tolerance.
        self.assertTrue(np.allclose(add_out, a0 + b0, atol=5e-2, rtol=5e-2))
        self.assertTrue(np.allclose(mul_out, a0 * b0, atol=5e-2, rtol=5e-2))
        self.assertTrue(np.allclose(mm_out, a0 @ b0, atol=1e-1, rtol=1e-1))
        self.assertTrue(np.allclose(sm_out, (1.5 - 0.25j) * a0, atol=5e-2, rtol=5e-2))

    def test_persistence_roundtrip_complex_matrices(self):
        cases = [
            (pycauset.complex_float16, "ComplexFloat16Matrix", np.complex64, 1e-2),
            (pycauset.complex_float32, "ComplexFloat32Matrix", np.complex64, 0.0),
            (pycauset.complex_float64, "ComplexFloat64Matrix", np.complex128, 0.0),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            for dtype_token, clsname, np_dtype, atol in cases:
                cls = getattr(pycauset, clsname, None)
                if cls is None:
                    continue

                m = pycauset.Matrix(2, dtype=dtype_token)
                m2 = None
                try:
                    m[0, 0] = 1 + 2j
                    m[1, 1] = 3 - 4j

                    path = str(Path(tmp) / f"persist_{dtype_token}.zip")
                    pycauset.save(m, path)
                    m2 = pycauset.load(path)

                    a = np.array(m).astype(np_dtype)
                    b = np.array(m2)
                    self.assertEqual(b.dtype, np.dtype(np_dtype))
                    self.assertTrue(np.allclose(a, b, atol=atol, rtol=0.0))
                finally:
                    if hasattr(m, "close"):
                        try:
                            m.close()
                        except Exception:
                            pass
                    if m2 is not None and hasattr(m2, "close"):
                        try:
                            m2.close()
                        except Exception:
                            pass


if __name__ == "__main__":
    unittest.main()
