import unittest

import pycauset


class TestPropertiesDispatch(unittest.TestCase):
    def test_matmul_diagonal_properties_drive_dispatch(self):
        a = pycauset.FloatMatrix(2)
        b = pycauset.FloatMatrix(2)
        try:
            # Note: off-diagonal entries are nonzero, but properties are gospel.
            a.set(0, 0, 1.0)
            a.set(0, 1, 100.0)
            a.set(1, 0, 200.0)
            a.set(1, 1, 4.0)

            b.set(0, 0, 5.0)
            b.set(0, 1, 6.0)
            b.set(1, 0, 7.0)
            b.set(1, 1, 8.0)

            a.properties = {"is_diagonal": True}

            pycauset._debug_clear_kernel_trace()
            c = pycauset.matmul(a, b)
            try:
                tag = pycauset._debug_last_kernel_trace()
                self.assertIn("diag_x_dense", tag)

                # Expected semantics: treat A as diagonal using only its diagonal.
                self.assertEqual(c.get(0, 0), 1.0 * 5.0)
                self.assertEqual(c.get(0, 1), 1.0 * 6.0)
                self.assertEqual(c.get(1, 0), 4.0 * 7.0)
                self.assertEqual(c.get(1, 1), 4.0 * 8.0)
            finally:
                c.close()
        finally:
            a.close()
            b.close()

    def test_matmul_triangular_properties_drive_dispatch(self):
        a = pycauset.FloatMatrix(3)
        b = pycauset.FloatMatrix(3)
        try:
            # Fill A and B with values (including below-diagonal noise).
            vals_a = [
                [1.0, 2.0, 3.0],
                [9.0, 4.0, 5.0],
                [8.0, 7.0, 6.0],
            ]
            vals_b = [
                [1.0, 1.0, 1.0],
                [2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0],
            ]

            for i in range(3):
                for j in range(3):
                    a.set(i, j, vals_a[i][j])
                    b.set(i, j, vals_b[i][j])

            a.properties = {"is_upper_triangular": True}
            b.properties = {"is_upper_triangular": True}

            pycauset._debug_clear_kernel_trace()
            c = pycauset.matmul(a, b)
            try:
                tag = pycauset._debug_last_kernel_trace()
                self.assertIn("tri_f64", tag)

                # Expected semantics: treat inputs as upper triangular.
                def a_eff(i: int, j: int) -> float:
                    return vals_a[i][j] if j >= i else 0.0

                def b_eff(i: int, j: int) -> float:
                    return vals_b[i][j] if j >= i else 0.0

                expected = [[0.0 for _ in range(3)] for _ in range(3)]
                for i in range(3):
                    for j in range(3):
                        s = 0.0
                        for k in range(3):
                            s += a_eff(i, k) * b_eff(k, j)
                        expected[i][j] = s

                for i in range(3):
                    for j in range(3):
                        self.assertEqual(c.get(i, j), expected[i][j])
            finally:
                c.close()
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
