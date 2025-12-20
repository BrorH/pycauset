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
            "div": "divide",
            "matmul": "matmul",
            "matvec": "matvec",
            "vecmat": "vecmat",
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

        def scalar_value(token: str):
            if token == "scalar_int64":
                return 3
            if token == "scalar_float64":
                return 1.5
            if token == "scalar_complex128":
                return 1.25 - 0.5j
            raise ValueError(f"Unknown scalar token: {token}")

        def expected_vector_scalar_result_dtype(v_dtype: str, scalar_token: str, *, op: str) -> str:
            # Mirrors the C++ implementation in src/math/LinearAlgebra.cpp:
            # - Complex vectors preserve complex dtype for real/complex scalar mul, and for real scalar add.
            # - For non-complex vectors:
            #    - float scalar: float16/float32 preserved; everything else -> float64
            #    - int scalar: int16 preserved; int32/bit -> int32; float16/float32 preserved; everything else -> float64
            if v_dtype.startswith("complex_"):
                if scalar_token == "scalar_complex128" and op != "mul_scalar":
                    raise ValueError("Complex scalar add is not supported")
                return v_dtype

            if scalar_token == "scalar_complex128":
                raise ValueError("Complex scalar requires complex vector")

            if scalar_token == "scalar_float64":
                if v_dtype == "float16":
                    return "float16"
                if v_dtype == "float32":
                    return "float32"
                return "float64"

            if scalar_token == "scalar_int64":
                if v_dtype == "int16":
                    return "int16"
                if v_dtype in ("int32", "bit"):
                    return "int32"
                if v_dtype == "float16":
                    return "float16"
                if v_dtype == "float32":
                    return "float32"
                return "float64"

            raise ValueError(f"Unknown scalar token: {scalar_token}")

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

        def values_matrix_denominator(dtype: str) -> list[list[object]]:
            # Avoid zeros in denominators. The NumPy reference path uses complex128
            # arrays for uniformity, and complex division by 0 produces NaNs.
            if dtype == "bit" or dtype.startswith("uint"):
                return [[1, 1], [1, 1]]
            # Current defaults for other dtypes contain no zeros.
            return values_matrix(dtype)

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

        def make_matrix(dtype: str, *, for_divisor: bool = False):
            m = pycauset.empty((2, 2), dtype=dtype)
            vals = values_matrix_denominator(dtype) if for_divisor else values_matrix(dtype)
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
                        b = make_matrix(b_dtype, for_divisor=(case.op == "div"))
                        out = None
                        try:
                            a_np = np.array(values_matrix(a_dtype), dtype=np.complex128)
                            b_np = np.array(
                                values_matrix_denominator(b_dtype) if case.op == "div" else values_matrix(b_dtype),
                                dtype=np.complex128,
                            )

                            if case.op == "add":
                                out = a + b
                                expected = a_np + b_np
                            elif case.op == "sub":
                                out = a - b
                                expected = a_np - b_np
                            elif case.op == "mul":
                                out = a * b
                                expected = a_np * b_np
                            elif case.op == "div":
                                out = a / b
                                expected = a_np / b_np
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

                    elif case.kind == "vecmat":
                        v = make_vector(a_dtype)
                        m = make_matrix(b_dtype)
                        out = None
                        try:
                            v_np = np.array(values_vector(a_dtype), dtype=np.complex128)
                            m_np = np.array(values_matrix(b_dtype), dtype=np.complex128)
                            out = v @ m
                            expected = (v_np @ m_np).reshape(1, -1)
                            res = pycauset._debug_resolve_promotion(op_to_resolver[case.op], a_dtype, b_dtype)
                            res_dtype = res["result_dtype"]
                            out_np = np.array(out)
                            expected_q = quantize_expected(expected, res_dtype)
                            self.assertEqual(out_np.dtype, np.dtype(dtype_to_np[res_dtype]), f"{label}: dtype mismatch")
                            assert_close(label, out_np, expected_q, dtype_token=res_dtype)
                        finally:
                            for obj in (out, m, v):
                                if obj is not None and hasattr(obj, "close"):
                                    try:
                                        obj.close()
                                    except Exception:
                                        pass

                    elif case.kind == "vector_scalar":
                        v = make_vector(a_dtype)
                        s_token = b_dtype
                        s = scalar_value(s_token)

                        out1 = None
                        out2 = None
                        try:
                            v_np = np.array(values_vector(a_dtype), dtype=np.complex128)

                            if case.op == "add_scalar":
                                out1 = v + s
                                out2 = s + v
                                expected = v_np + s
                                res_dtype = expected_vector_scalar_result_dtype(a_dtype, s_token, op="add_scalar")
                            elif case.op == "mul_scalar":
                                out1 = v * s
                                out2 = s * v
                                expected = v_np * s
                                res_dtype = expected_vector_scalar_result_dtype(a_dtype, s_token, op="mul_scalar")
                            else:
                                raise ValueError(f"Unknown op: {case.op}")

                            out1_np = np.array(out1)
                            out2_np = np.array(out2)

                            expected_q = quantize_expected(expected, res_dtype)
                            self.assertEqual(out1_np.dtype, np.dtype(dtype_to_np[res_dtype]), f"{label}: dtype mismatch")
                            self.assertEqual(out2_np.dtype, np.dtype(dtype_to_np[res_dtype]), f"{label}: dtype mismatch")
                            assert_close(label + ":lhs", out1_np, expected_q, dtype_token=res_dtype)
                            assert_close(label + ":rhs", out2_np, expected_q, dtype_token=res_dtype)
                        finally:
                            for obj in (out2, out1, v):
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
