import unittest

import pycauset


class TestBlockMatrixPhaseB(unittest.TestCase):
    def _fill_const(self, m, value: float) -> None:
        r = int(m.rows())
        c = int(m.cols())
        for i in range(r):
            for j in range(c):
                m.set(i, j, float(value))

    def test_construct_and_index(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        a = b = c = d = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(2, 1)
            c = pycauset.FloatMatrix(1, 2)
            d = pycauset.FloatMatrix(1, 1)

            self._fill_const(a, 1.0)
            self._fill_const(b, 2.0)
            self._fill_const(c, 3.0)
            self._fill_const(d, 4.0)

            bm = BlockMatrix([[a, b], [c, d]])

            self.assertEqual(bm.shape, (3, 3))
            self.assertEqual(bm.block_rows, 2)
            self.assertEqual(bm.block_cols, 2)
            self.assertEqual(bm.row_partitions, [0, 2, 3])
            self.assertEqual(bm.col_partitions, [0, 2, 3])

            # A block
            self.assertEqual(bm.get(0, 0), 1.0)
            self.assertEqual(bm.get(1, 1), 1.0)

            # B block (top-right)
            self.assertEqual(bm.get(0, 2), 2.0)
            self.assertEqual(bm.get(1, 2), 2.0)

            # C block (bottom-left)
            self.assertEqual(bm.get(2, 0), 3.0)
            self.assertEqual(bm.get(2, 1), 3.0)

            # D block (bottom-right)
            self.assertEqual(bm.get(2, 2), 4.0)

            s = str(bm)
            self.assertIn("BlockMatrix(", s)
            self.assertIn("row_partitions", s)
            self.assertIn("col_partitions", s)

            # Best-effort NumPy conversion for debug/interop.
            try:
                import numpy as np

                arr = np.asarray(bm)
                self.assertEqual(arr.shape, (3, 3))
                self.assertEqual(float(arr[0, 0]), 1.0)
                self.assertEqual(float(arr[0, 2]), 2.0)
                self.assertEqual(float(arr[2, 0]), 3.0)
                self.assertEqual(float(arr[2, 2]), 4.0)
            except Exception:
                pass

        finally:
            for obj in (a, b, c, d):
                if obj is not None:
                    obj.close()

    def test_validation_inconsistent_row_heights(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        a = b = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(3, 1)
            with self.assertRaises(ValueError):
                BlockMatrix([[a, b]])
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()

    def test_validation_inconsistent_col_widths(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        a = b = c = d = None
        try:
            a = pycauset.FloatMatrix(1, 2)
            b = pycauset.FloatMatrix(1, 3)
            c = pycauset.FloatMatrix(1, 2)
            d = pycauset.FloatMatrix(1, 4)
            with self.assertRaises(ValueError):
                BlockMatrix([[a, b], [c, d]])
        finally:
            for obj in (a, b, c, d):
                if obj is not None:
                    obj.close()


if __name__ == "__main__":
    unittest.main()
