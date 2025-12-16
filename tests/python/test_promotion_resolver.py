import unittest

import pycauset


class TestPromotionResolver(unittest.TestCase):
    def test_all_current_dtypes_resolve_for_all_ops(self):
        ops = [
            "add",
            "subtract",
            "elementwise_multiply",
            "matmul",
            "matvec",
            "vecmat",
            "outer_product",
        ]
        dtypes = [
            "bit",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "complex_float16",
            "complex_float32",
            "complex_float64",
        ]

        signed_ints = {"int8", "int16", "int32", "int64"}

        def is_known_invalid_integer_mix(a: str, b: str) -> bool:
            # C++ promotion rule: mixed signed/unsigned requires a signed type with
            # signed_bits >= unsigned_bits + 1. With uint64 there is no such signed type.
            return (a == "uint64" and b in signed_ints) or (b == "uint64" and a in signed_ints)

        for op in ops:
            for a in dtypes:
                for b in dtypes:
                    if is_known_invalid_integer_mix(a, b) and op in (
                        "add",
                        "subtract",
                        "elementwise_multiply",
                        "matmul",
                        "matvec",
                        "vecmat",
                        "outer_product",
                    ):
                        with self.assertRaises(Exception):
                            pycauset._debug_resolve_promotion(op, a, b)
                        continue

                    d = pycauset._debug_resolve_promotion(op, a, b)
                    self.assertIn(d["result_dtype"], dtypes)
                    self.assertIn(d["float_underpromotion"], (True, False))
                    # chosen_float_dtype is only meaningful for mixed-float underpromotion.
                    self.assertIn(d["chosen_float_dtype"], ["unknown", "float16", "float32", "float64"])

    def test_matmul_mixed_float_underpromotes_to_float32(self):
        d = pycauset._debug_resolve_promotion("matmul", "float32", "float64")
        self.assertEqual(d["result_dtype"], "float32")
        self.assertEqual(d["float_underpromotion"], True)
        self.assertEqual(d["chosen_float_dtype"], "float32")

        d2 = pycauset._debug_resolve_promotion("matmul", "float64", "float32")
        self.assertEqual(d2["result_dtype"], "float32")
        self.assertEqual(d2["float_underpromotion"], True)

    def test_add_kind_rules(self):
        self.assertEqual(pycauset._debug_resolve_promotion("add", "bit", "bit")["result_dtype"], "int32")
        self.assertEqual(pycauset._debug_resolve_promotion("add", "bit", "int32")["result_dtype"], "int32")
        self.assertEqual(pycauset._debug_resolve_promotion("add", "int32", "bit")["result_dtype"], "int32")
        self.assertEqual(pycauset._debug_resolve_promotion("add", "bit", "float32")["result_dtype"], "float32")

        # int16 is preserved when both operands are int16.
        self.assertEqual(pycauset._debug_resolve_promotion("add", "int16", "int16")["result_dtype"], "int16")
        # bit promotes to the integer dtype when mixed with integers.
        self.assertEqual(pycauset._debug_resolve_promotion("add", "bit", "int16")["result_dtype"], "int16")
        self.assertEqual(pycauset._debug_resolve_promotion("add", "int16", "bit")["result_dtype"], "int16")

    def test_elementwise_multiply_bit_bit_stays_bit(self):
        self.assertEqual(
            pycauset._debug_resolve_promotion("elementwise_multiply", "bit", "bit")["result_dtype"],
            "bit",
        )

    def test_matvec_mixed_float_underpromotes_to_float32(self):
        d = pycauset._debug_resolve_promotion("matvec", "float32", "float64")
        self.assertEqual(d["result_dtype"], "float32")
        self.assertEqual(d["float_underpromotion"], True)
        self.assertEqual(d["chosen_float_dtype"], "float32")

    def test_matmul_mixed_complex_underpromotes_to_complex_float32(self):
        d = pycauset._debug_resolve_promotion("matmul", "complex_float32", "complex_float64")
        self.assertEqual(d["result_dtype"], "complex_float32")
        # Underpromotion warnings are currently only tracked for real floats.
        self.assertEqual(d["float_underpromotion"], False)
        self.assertEqual(d["chosen_float_dtype"], "unknown")

        d2 = pycauset._debug_resolve_promotion("matmul", "complex_float64", "complex_float32")
        self.assertEqual(d2["result_dtype"], "complex_float32")
        self.assertEqual(d2["float_underpromotion"], False)


if __name__ == "__main__":
    unittest.main()
