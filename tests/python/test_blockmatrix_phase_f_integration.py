import unittest

import pycauset


class TestBlockMatrixPhaseFIntegration(unittest.TestCase):
    def _fill_const(self, m, value: float) -> None:
        fill = getattr(m, "fill", None)
        if callable(fill):
            fill(float(value))
            return
        r = int(m.rows())
        c = int(m.cols())
        for i in range(r):
            for j in range(c):
                m.set(i, j, float(value))

    def test_matrix_constructor_block_grid_returns_blockmatrix(self):
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

            bm = pycauset.matrix([[a, b], [c, d]])
            self.assertIsInstance(bm, BlockMatrix)
            self.assertEqual(bm.shape, (3, 3))
            self.assertEqual(bm.get(0, 0), 1.0)
            self.assertEqual(bm.get(0, 2), 2.0)
            self.assertEqual(bm.get(2, 0), 3.0)
            self.assertEqual(bm.get(2, 2), 4.0)
        finally:
            for obj in (a, b, c, d):
                if obj is not None:
                    obj.close()

    def test_matrix_constructor_mixed_matrices_and_scalars_raises(self):
        a = None
        try:
            a = pycauset.FloatMatrix(1, 1)
            self._fill_const(a, 1.0)
            with self.assertRaises(TypeError):
                _ = pycauset.matrix([[a, 1.0], [2.0, 3.0]])
        finally:
            if a is not None:
                a.close()

    def test_matmul_once_block_always_block_and_lazy(self):
        from pycauset._internal.blockmatrix import BlockMatrix
        from pycauset._internal.thunks import ThunkBlock

        a = b = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(2, 2)
            self._fill_const(a, 2.0)
            self._fill_const(b, 3.0)

            left = BlockMatrix([[a]])
            right = BlockMatrix([[b]])

            pycauset._debug_clear_kernel_trace()
            out = pycauset.matmul(left, right)

            self.assertIsInstance(out, BlockMatrix)
            self.assertIsInstance(out.get_block(0, 0), ThunkBlock)

            # Construction must not evaluate leaf kernels.
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")

            # Trigger evaluation.
            _ = out.get(0, 0)
            self.assertNotEqual(pycauset._debug_last_kernel_trace(), "")
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()

    def test_dense_matmul_block_uses_rmatmul_fallback(self):
        from pycauset._internal.blockmatrix import BlockMatrix
        from pycauset._internal.thunks import ThunkBlock

        a = b = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(2, 2)
            self._fill_const(a, 2.0)
            self._fill_const(b, 3.0)

            block = BlockMatrix([[b]])

            pycauset._debug_clear_kernel_trace()
            out = a @ block
            self.assertIsInstance(out, BlockMatrix)
            self.assertIsInstance(out.get_block(0, 0), ThunkBlock)
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out.get(0, 0)
            self.assertNotEqual(pycauset._debug_last_kernel_trace(), "")
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()

    def test_dense_add_block_uses_radd_fallback(self):
        from pycauset._internal.blockmatrix import BlockMatrix
        from pycauset._internal.thunks import ThunkBlock

        a = b = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(2, 2)
            self._fill_const(a, 2.0)
            self._fill_const(b, 3.0)

            block = BlockMatrix([[b]])

            pycauset._debug_clear_kernel_trace()
            out = a + block
            self.assertIsInstance(out, BlockMatrix)
            self.assertIsInstance(out.get_block(0, 0), ThunkBlock)
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out.get(0, 0)
            self.assertNotEqual(pycauset._debug_last_kernel_trace(), "")
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()

    def test_dense_sub_block_uses_rsub_fallback(self):
        from pycauset._internal.blockmatrix import BlockMatrix
        from pycauset._internal.thunks import ThunkBlock

        a = b = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(2, 2)
            self._fill_const(a, 2.0)
            self._fill_const(b, 3.0)

            block = BlockMatrix([[b]])

            pycauset._debug_clear_kernel_trace()
            out = a - block
            self.assertIsInstance(out, BlockMatrix)
            self.assertIsInstance(out.get_block(0, 0), ThunkBlock)
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out.get(0, 0)
            self.assertNotEqual(pycauset._debug_last_kernel_trace(), "")
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()

    def test_dense_mul_block_uses_rmul_fallback(self):
        from pycauset._internal.blockmatrix import BlockMatrix
        from pycauset._internal.thunks import ThunkBlock

        a = b = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(2, 2)
            self._fill_const(a, 2.0)
            self._fill_const(b, 3.0)

            block = BlockMatrix([[b]])

            pycauset._debug_clear_kernel_trace()
            out = a * block
            self.assertIsInstance(out, BlockMatrix)
            self.assertIsInstance(out.get_block(0, 0), ThunkBlock)
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out.get(0, 0)
            self.assertNotEqual(pycauset._debug_last_kernel_trace(), "")
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()

    def test_dense_div_block_uses_rtruediv_fallback(self):
        from pycauset._internal.blockmatrix import BlockMatrix
        from pycauset._internal.thunks import ThunkBlock

        a = b = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(2, 2)
            self._fill_const(a, 6.0)
            self._fill_const(b, 3.0)

            block = BlockMatrix([[b]])

            pycauset._debug_clear_kernel_trace()
            out = a / block
            self.assertIsInstance(out, BlockMatrix)
            self.assertIsInstance(out.get_block(0, 0), ThunkBlock)
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out.get(0, 0)
            self.assertNotEqual(pycauset._debug_last_kernel_trace(), "")
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()

    def test_property_aware_leaf_dispatch_is_used_inside_thunks(self):
        from pycauset._internal.blockmatrix import BlockMatrix
        from pycauset._internal import properties as _props

        a = b = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(2, 2)

            # Make 'a' diagonal (payload) and assert diagonal (properties).
            a.set(0, 0, 2.0)
            a.set(0, 1, 0.0)
            a.set(1, 0, 0.0)
            a.set(1, 1, 5.0)
            _props.set_properties(a, {"is_diagonal": True})

            self._fill_const(b, 3.0)

            left = BlockMatrix([[a]])
            right = BlockMatrix([[b]])

            pycauset._debug_clear_kernel_trace()
            out = pycauset.matmul(left, right)
            _ = out.get(0, 0)

            trace = pycauset._debug_last_kernel_trace()
            self.assertIn("matmul", trace)
            self.assertIn("diag_x_dense", trace)
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()

    def test_io_prefetch_discard_trace_emitted_during_thunk_eval(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        a = b = None
        try:
            a = pycauset.FloatMatrix(2, 2)
            b = pycauset.FloatMatrix(2, 2)
            self._fill_const(a, 2.0)
            self._fill_const(b, 3.0)

            left = BlockMatrix([[a]])
            right = BlockMatrix([[b]])

            pycauset._debug_clear_io_trace()
            out = pycauset.matmul(left, right)
            _ = out.get(0, 0)

            self.assertEqual(pycauset._debug_last_io_trace(), "io.discard")
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()

if __name__ == "__main__":
    unittest.main()
