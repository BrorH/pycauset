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

    def test_device_routing_block_add_under_cuda(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        # Enable CUDA if available; skip if the plugin is not present/active.
        if not hasattr(pycauset, "cuda"):
            self.skipTest("pycauset.cuda is not available")

        pycauset.cuda.enable()
        if not pycauset.cuda.is_available():
            self.skipTest("CUDA is not active in this build/environment")

        n = 600  # exceeds AutoSolver gpu_threshold_elements_ (512*512)
        a32 = b32 = a16 = b16 = None
        try:
            a32 = pycauset.Float32Matrix(n, n)
            b32 = pycauset.Float32Matrix(n, n)
            self._fill_const(a32, 1.0)
            self._fill_const(b32, 2.0)

            left32 = BlockMatrix([[a32]])
            right32 = BlockMatrix([[b32]])

            pycauset._debug_clear_kernel_trace()
            out32 = left32 + right32
            # Construction must not run kernels.
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out32.get(0, 0)
            trace = pycauset._debug_last_kernel_trace()
            self.assertIn("gpu.add.f32", trace)

            # Float16 add is not supported on CUDA path (AutoSolver), so it must fall back to CPU.
            a16 = pycauset.Float16Matrix(n, n)
            b16 = pycauset.Float16Matrix(n, n)
            self._fill_const(a16, 1.0)
            self._fill_const(b16, 2.0)

            left16 = BlockMatrix([[a16]])
            right16 = BlockMatrix([[b16]])

            pycauset._debug_clear_kernel_trace()
            out16 = left16 + right16
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out16.get(0, 0)
            trace = pycauset._debug_last_kernel_trace()
            self.assertIn("cpu.add.f16", trace)
        finally:
            for obj in (a32, b32, a16, b16):
                if obj is not None:
                    obj.close()
            try:
                pycauset.cuda.disable()
            except Exception:
                pass

    def test_device_routing_block_sub_under_cuda(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        if not hasattr(pycauset, "cuda"):
            self.skipTest("pycauset.cuda is not available")

        pycauset.cuda.enable()
        if not pycauset.cuda.is_available():
            self.skipTest("CUDA is not active in this build/environment")

        n = 600
        a32 = b32 = a16 = b16 = None
        try:
            a32 = pycauset.Float32Matrix(n, n)
            b32 = pycauset.Float32Matrix(n, n)
            self._fill_const(a32, 4.0)
            self._fill_const(b32, 1.0)

            left32 = BlockMatrix([[a32]])
            right32 = BlockMatrix([[b32]])

            pycauset._debug_clear_kernel_trace()
            out32 = left32 - right32
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out32.get(0, 0)
            trace = pycauset._debug_last_kernel_trace()
            self.assertIn("gpu.subtract.f32", trace)

            a16 = pycauset.Float16Matrix(n, n)
            b16 = pycauset.Float16Matrix(n, n)
            self._fill_const(a16, 4.0)
            self._fill_const(b16, 1.0)

            left16 = BlockMatrix([[a16]])
            right16 = BlockMatrix([[b16]])

            pycauset._debug_clear_kernel_trace()
            out16 = left16 - right16
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out16.get(0, 0)
            trace = pycauset._debug_last_kernel_trace()
            self.assertIn("cpu.subtract.f16", trace)
        finally:
            for obj in (a32, b32, a16, b16):
                if obj is not None:
                    obj.close()
            try:
                pycauset.cuda.disable()
            except Exception:
                pass

    def test_cpu_only_ops_under_cuda(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        if not hasattr(pycauset, "cuda"):
            self.skipTest("pycauset.cuda is not available")

        pycauset.cuda.enable()
        if not pycauset.cuda.is_available():
            self.skipTest("CUDA is not active in this build/environment")

        n = 600
        a = b = None
        try:
            a = pycauset.Float32Matrix(n, n)
            b = pycauset.Float32Matrix(n, n)
            self._fill_const(a, 2.0)
            self._fill_const(b, 4.0)

            left = BlockMatrix([[a]])
            right = BlockMatrix([[b]])

            # CUDA elementwise_multiply is not implemented -> CPU-only
            pycauset._debug_clear_kernel_trace()
            out_mul = left * right
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out_mul.get(0, 0)
            self.assertIn("cpu.mul.f32", pycauset._debug_last_kernel_trace())

            # CUDA elementwise_divide is not implemented -> CPU-only
            pycauset._debug_clear_kernel_trace()
            out_div = left / right
            self.assertEqual(pycauset._debug_last_kernel_trace(), "")
            _ = out_div.get(0, 0)
            self.assertIn("cpu.div.f32", pycauset._debug_last_kernel_trace())
        finally:
            for obj in (a, b):
                if obj is not None:
                    obj.close()
            try:
                pycauset.cuda.disable()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
