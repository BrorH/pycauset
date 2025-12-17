import unittest
import warnings


class TestSupportMatrix(unittest.TestCase):
    def test_support_matrix_cases(self):
        import pycauset
        from pycauset._internal.support_matrix import SUPPORTED

        import numpy as np

        failures: list[str] = []

        op_to_resolver = {
            "add": "add",
            "sub": "subtract",
            "mul": "elementwise_multiply",
            "matmul": "matmul",
            "matvec": "matvec",
            "outer": "outer_product",
        }

        dtype_to_np = {
            "bit": np.bool_,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "complex_float16": np.complex64,
            "complex_float32": np.complex64,
            "complex_float64": np.complex128,
        }

        def quantize_expected(expected: np.ndarray, result_dtype: str) -> np.ndarray:
            if result_dtype == "complex_float16":
                re = expected.real.astype(np.float16).astype(np.float32)
                im = expected.imag.astype(np.float16).astype(np.float32)
                return (re + 1j * im).astype(np.complex64)

            np_dt = dtype_to_np[result_dtype]
            if np_dt is np.bool_:
                return (expected != 0)
            return expected.astype(np_dt)

        def assert_close(label: str, got, expected, *, dtype_token: str):
            if dtype_token in ("float16", "complex_float16"):
                atol, rtol = 5e-2, 5e-2
            elif dtype_token in ("float32", "complex_float32"):
                atol, rtol = 1e-6, 1e-6
            else:
                atol, rtol = 0.0, 0.0

            if isinstance(got, np.ndarray):
                if got.dtype == np.bool_ or expected.dtype == np.bool_:
                    self.assertTrue(np.array_equal(got, expected), f"{label}: boolean mismatch")
                    return
                if atol == 0.0 and rtol == 0.0 and np.issubdtype(got.dtype, np.integer):
                    self.assertTrue(np.array_equal(got, expected), f"{label}: integer mismatch")
                    return
                self.assertTrue(np.allclose(got, expected, atol=atol, rtol=rtol), f"{label}: mismatch")
                return

            self.assertTrue(np.allclose(np.array(got), np.array(expected), atol=atol, rtol=rtol), f"{label}: scalar mismatch")

        def values_matrix(dtype: str) -> list[list[object]]:
            if dtype == "bit":
                return [[1, 0], [1, 1]]
            if dtype.startswith("uint"):
                # Keep unsigned test values in {0,1} so subtraction never goes negative
                # when the promoted output dtype is also unsigned.
                return [[1, 0], [1, 1]]
            if dtype in ("int8", "int16", "int32", "int64"):
                return [[2, -3], [4, 5]]
            if dtype in ("float16", "float32", "float64"):
                return [[1.25, -2.5], [3.0, 0.5]]
            return [[1 + 2j, 3 - 4j], [-5 + 0.5j, 0 + 6j]]

        def values_vector(dtype: str) -> list[object]:
            if dtype == "bit":
                return [1, 0]
            if dtype.startswith("uint"):
                return [1, 0]
            if dtype in ("int8", "int16", "int32", "int64"):
                return [2, -3]
            if dtype in ("float16", "float32", "float64"):
                return [1.25, -2.5]
            return [1 + 2j, -3 + 0.5j]

        def make_matrix(dtype: str):
            m = pycauset.empty((2, 2), dtype=dtype)
            vals = values_matrix(dtype)
            m[0, 0] = vals[0][0]
            m[0, 1] = vals[0][1]
            m[1, 0] = vals[1][0]
            m[1, 1] = vals[1][1]
            return m

        def make_vector(dtype: str):
            v = pycauset.empty(2, dtype=dtype)
            vals = values_vector(dtype)
            v[0] = vals[0]
            v[1] = vals[1]
            return v

        for case in SUPPORTED:
            a_dtype = case.a_dtype
            b_dtype = case.b_dtype or case.a_dtype
            label = f"{case.kind}:{case.op}:{a_dtype},{b_dtype}"

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if case.kind == "matrix":
                        a = make_matrix(a_dtype)
                        b = make_matrix(b_dtype)
                        out = None
                        try:
                            a_np = np.array(values_matrix(a_dtype), dtype=np.complex128)
                            b_np = np.array(values_matrix(b_dtype), dtype=np.complex128)

                            if case.op == "add":
                                out = a + b
                                expected = a_np + b_np
                            elif case.op == "sub":
                                out = a - b
                                expected = a_np - b_np
                            elif case.op == "mul":
                                out = a * b
                                expected = a_np * b_np
                            elif case.op == "matmul":
                                out = a @ b
                                expected = a_np @ b_np
                            elif case.op == "H":
                                out = a.H
                                expected = a_np.T.conj()
                            elif case.op == "conj":
                                out = a.conj()
                                expected = np.conj(a_np)
                            else:
                                raise ValueError(f"Unknown op: {case.op}")

                            out_np = np.array(out)
                            if case.op in ("H", "conj"):
                                res_dtype = a_dtype
                            else:
                                res = pycauset._debug_resolve_promotion(op_to_resolver[case.op], a_dtype, b_dtype)
                                res_dtype = res["result_dtype"]

                            expected_q = quantize_expected(expected, res_dtype)
                            self.assertEqual(out_np.dtype, np.dtype(dtype_to_np[res_dtype]), f"{label}: dtype mismatch")
                            assert_close(label, out_np, expected_q, dtype_token=res_dtype)
                        finally:
                            for obj in (out, b, a):
                                if obj is not None and hasattr(obj, "close"):
                                    try:
                                        obj.close()
                                    except Exception:
                                        pass

                    elif case.kind == "vector":
                        a = make_vector(a_dtype)
                        b = make_vector(b_dtype)
                        out = None
                        try:
                            a_np = np.array(values_vector(a_dtype), dtype=np.complex128)
                            b_np = np.array(values_vector(b_dtype), dtype=np.complex128)

                            if case.op == "add":
                                out = a + b
                                expected = a_np + b_np
                                res = pycauset._debug_resolve_promotion(op_to_resolver[case.op], a_dtype, b_dtype)
                                res_dtype = res["result_dtype"]
                                out_np = np.array(out)
                                expected_q = quantize_expected(expected, res_dtype)
                                self.assertEqual(out_np.dtype, np.dtype(dtype_to_np[res_dtype]), f"{label}: dtype mismatch")
                                assert_close(label, out_np, expected_q, dtype_token=res_dtype)
                            elif case.op == "sub":
                                out = a - b
                                expected = a_np - b_np
                                res = pycauset._debug_resolve_promotion(op_to_resolver[case.op], a_dtype, b_dtype)
                                res_dtype = res["result_dtype"]
                                out_np = np.array(out)
                                expected_q = quantize_expected(expected, res_dtype)
                                self.assertEqual(out_np.dtype, np.dtype(dtype_to_np[res_dtype]), f"{label}: dtype mismatch")
                                assert_close(label, out_np, expected_q, dtype_token=res_dtype)
                            elif case.op == "dot":
                                out = a.dot(b)
                                expected = (a_np * b_np).sum()
                                if a_dtype.startswith("complex") or b_dtype.startswith("complex"):
                                    assert_close(label, out, expected, dtype_token="complex_float64")
                                else:
                                    assert_close(label, float(out), float(expected.real), dtype_token="float64")
                            elif case.op == "outer":
                                out = a @ b.T
                                expected = np.outer(a_np, b_np)
                                res = pycauset._debug_resolve_promotion(op_to_resolver[case.op], a_dtype, b_dtype)
                                res_dtype = res["result_dtype"]
                                out_np = np.array(out)
                                expected_q = quantize_expected(expected, res_dtype)
                                self.assertEqual(out_np.dtype, np.dtype(dtype_to_np[res_dtype]), f"{label}: dtype mismatch")
                                assert_close(label, out_np, expected_q, dtype_token=res_dtype)
                            elif case.op == "H":
                                out = a.H
                                # Vectors behave as column vectors; H produces a 1xN row view.
                                expected = np.conj(a_np).reshape(1, -1)
                                out_np = np.array(out)
                                expected_q = quantize_expected(expected, a_dtype)
                                self.assertEqual(out_np.dtype, np.dtype(dtype_to_np[a_dtype]), f"{label}: dtype mismatch")
                                assert_close(label, out_np, expected_q, dtype_token=a_dtype)
                            elif case.op == "conj":
                                out = a.conj()
                                expected = np.conj(a_np)
                                out_np = np.array(out)
                                expected_q = quantize_expected(expected, a_dtype)
                                self.assertEqual(out_np.dtype, np.dtype(dtype_to_np[a_dtype]), f"{label}: dtype mismatch")
                                assert_close(label, out_np, expected_q, dtype_token=a_dtype)
                            else:
                                raise ValueError(f"Unknown op: {case.op}")
                        finally:
                            for obj in (out, b, a):
                                if obj is not None and hasattr(obj, "close"):
                                    try:
                                        obj.close()
                                    except Exception:
                                        pass

                    elif case.kind == "matvec":
                        m = make_matrix(a_dtype)
                        v = make_vector(b_dtype)
                        out = None
                        try:
                            m_np = np.array(values_matrix(a_dtype), dtype=np.complex128)
                            v_np = np.array(values_vector(b_dtype), dtype=np.complex128)
                            out = m @ v
                            expected = m_np @ v_np
                            res = pycauset._debug_resolve_promotion(op_to_resolver[case.op], a_dtype, b_dtype)
                            res_dtype = res["result_dtype"]
                            out_np = np.array(out)
                            expected_q = quantize_expected(expected, res_dtype)
                            self.assertEqual(out_np.dtype, np.dtype(dtype_to_np[res_dtype]), f"{label}: dtype mismatch")
                            assert_close(label, out_np, expected_q, dtype_token=res_dtype)
                        finally:
                            for obj in (out, v, m):
                                if obj is not None and hasattr(obj, "close"):
                                    try:
                                        obj.close()
                                    except Exception:
                                        pass

                    else:
                        raise ValueError(f"Unknown kind: {case.kind}")

            except Exception as exc:
                failures.append(f"{label} -> {type(exc).__name__}: {exc}")

        if failures:
            self.fail("Support matrix regressions:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
