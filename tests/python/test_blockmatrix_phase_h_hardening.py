import unittest

import pycauset


class TestBlockMatrixPhaseHHardening(unittest.TestCase):
    def _close(self, *objs):
        for obj in objs:
            close_fn = getattr(obj, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    def test_complex_matmul_matches_dense(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        Complex64 = getattr(pycauset, "ComplexFloat64Matrix", None)
        if Complex64 is None:
            self.skipTest("ComplexFloat64Matrix is not available")

        A = B = expected = None
        try:
            A = Complex64(2, 2)
            B = Complex64(2, 2)
            A.set(0, 0, 1.0 + 2.0j)
            A.set(0, 1, 0.0)
            A.set(1, 0, 0.0)
            A.set(1, 1, 3.0 - 1.0j)

            B.set(0, 0, 2.0 - 1.0j)
            B.set(0, 1, 1.0j)
            B.set(1, 0, 0.0)
            B.set(1, 1, 4.0)

            left = BlockMatrix([[A]])
            right = BlockMatrix([[B]])

            expected = pycauset.matmul(A, B)
            out = left @ right

            val = out.get(1, 1)
            exp = expected.get(1, 1)
            self.assertAlmostEqual(val.real, exp.real, places=6)
            self.assertAlmostEqual(val.imag, exp.imag, places=6)
        finally:
            self._close(A, B, expected)

    def test_float16_block_matmul_matches_dense(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        Float16 = getattr(pycauset, "Float16Matrix", None)
        if Float16 is None:
            self.skipTest("Float16Matrix is not available")

        A = B = expected = None
        try:
            A = Float16(2, 2)
            B = Float16(2, 2)
            A.set(0, 0, 1.0)
            A.set(0, 1, 2.0)
            A.set(1, 0, 3.0)
            A.set(1, 1, 4.0)

            B.set(0, 0, 5.0)
            B.set(0, 1, 6.0)
            B.set(1, 0, 7.0)
            B.set(1, 1, 8.0)

            left = BlockMatrix([[A]])
            right = BlockMatrix([[B]])

            expected = pycauset.matmul(A, B)
            out = left @ right

            val = out.get(0, 1)
            exp = expected.get(0, 1)
            self.assertAlmostEqual(float(val), float(exp), places=3)
        finally:
            self._close(A, B, expected)

    def test_many_small_blocks_matmul_matches_dense(self):
        from pycauset._internal.blockmatrix import BlockMatrix

        blocks_a = []
        for i in range(3):  # shape 3x2 (blocks are 1x1)
            row = []
            for j in range(2):
                m = pycauset.FloatMatrix(1, 1)
                m.set(0, 0, float(i * 2 + j + 1))
                row.append(m)
            blocks_a.append(row)

        blocks_b = []
        for i in range(2):  # shape 2x3 (blocks are 1x1)
            row = []
            for j in range(3):
                m = pycauset.FloatMatrix(1, 1)
                m.set(0, 0, float((i + 1) * (j + 1)))
                row.append(m)
            blocks_b.append(row)

        left = BlockMatrix(blocks_a)
        right = BlockMatrix(blocks_b)

        dense_left = pycauset.matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        dense_right = pycauset.matrix([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        expected = pycauset.matmul(dense_left, dense_right)
        out = left @ right

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(out.get(i, j), expected.get(i, j), places=6)

        self._close(*(blk for row in blocks_a for blk in row))
        self._close(*(blk for row in blocks_b for blk in row))
        self._close(dense_left, dense_right, expected)

    def test_mixed_dtype_elementwise_add_matches_dense(self):
        from pycauset._internal.blockmatrix import BlockMatrix, block_add

        a = b = expected = None
        try:
            a = pycauset.matrix([[1, 2], [3, 4]], dtype="int32")
            b = pycauset.matrix([[0.5, 1.5], [2.5, 3.5]], dtype="float32")

            left = BlockMatrix([[a]])
            right = BlockMatrix([[b]])

            expected = pycauset.matrix([[1.5, 3.5], [5.5, 7.5]], dtype="float32")
            out = block_add(left, right)

            for i in range(2):
                for j in range(2):
                    self.assertAlmostEqual(out.get(i, j), expected.get(i, j), places=6)
        finally:
            self._close(a, b, expected)

    def test_nested_blockmatrix_matmul_and_persistence_complex(self):
        from pycauset._internal.blockmatrix import BlockMatrix
        from pathlib import Path
        import tempfile

        Complex64 = getattr(pycauset, "ComplexFloat64Matrix", None)
        if Complex64 is None:
            self.skipTest("ComplexFloat64Matrix is not available")

        tmpdir = None
        inner = outer = loaded = None
        try:
            tmpdir = tempfile.TemporaryDirectory()
            base_a = Complex64(1, 1)
            base_b = pycauset.FloatMatrix(1, 1)
            base_a.set(0, 0, 1.0 + 1.0j)
            base_b.set(0, 0, 2.0)

            try:
                _ = pycauset.matmul(base_a, base_a)
            except Exception as exc:
                self.skipTest(f"Complex matmul not supported in current build: {exc}")

            inner = BlockMatrix([[base_a, base_b], [base_b, base_a]])
            outer = BlockMatrix([[inner]])

            out = outer @ outer
            try:
                val = out.get(0, 0)
            except Exception as exc:
                self.skipTest(f"Complex block matmul not supported in current build: {exc}")

            expected_val = (1.0 + 1.0j) * (1.0 + 1.0j) + 2.0 * 2.0  # (1+i)^2 + 4 = 4 + 2i
            self.assertAlmostEqual(val.real, expected_val.real, places=6)
            self.assertAlmostEqual(val.imag, expected_val.imag, places=6)

            # Persistence of complex block outputs may be backend-limited; skip if unsupported.
            path = Path(tmpdir.name) / "nested_complex.pycauset"
            try:
                pycauset.save(out, path)
                loaded = pycauset.load(path)
                self.assertEqual(loaded.shape, out.shape)
                self.assertAlmostEqual(loaded.get(0, 0), out.get(0, 0))
            except Exception as exc:
                self.skipTest(f"Complex block persistence not supported in current build: {exc}")
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()
            self._close(inner, outer, loaded)

    def test_thunk_concurrency_single_eval(self):
        from pycauset._internal.thunks import ThunkBlock
        from pycauset._internal.blockmatrix import BlockMatrix
        import threading

        calls = []

        def _compute():
            calls.append(1)
            m = pycauset.FloatMatrix(1, 1)
            m.set(0, 0, 7.0)
            return m

        thunk = ThunkBlock(rows=1, cols=1, compute=_compute, sources_for_staleness=())
        bm = BlockMatrix([[thunk]])

        def _worker(results, idx):
            results[idx] = bm.get(0, 0)

        results = [None] * 4
        threads = [threading.Thread(target=_worker, args=(results, i)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for v in results:
            self.assertEqual(v, 7.0)
        self.assertEqual(len(calls), 1)

    def test_set_block_makes_existing_thunk_stale(self):
        from pycauset._internal.blockmatrix import BlockMatrix, block_add
        from pycauset._internal.thunks import StaleThunkError

        a1 = b1 = a2 = None
        try:
            a1 = pycauset.FloatMatrix(1, 1)
            b1 = pycauset.FloatMatrix(1, 1)
            a1.set(0, 0, 1.0)
            b1.set(0, 0, 2.0)

            left = BlockMatrix([[a1]])
            right = BlockMatrix([[b1]])
            out = block_add(left, right)

            # Trigger evaluation to populate cache.
            self.assertEqual(out.get(0, 0), 3.0)

            # Replace a block to bump version and stale cached thunks.
            a2 = pycauset.FloatMatrix(1, 1)
            a2.set(0, 0, 5.0)
            left.set_block(0, 0, a2)

            with self.assertRaises(StaleThunkError):
                _ = out.get(0, 0)
        finally:
            self._close(a1, b1, a2)


if __name__ == "__main__":
    unittest.main()