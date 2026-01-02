import os
import tempfile
import unittest

import pycauset


class TestIOObservability(unittest.TestCase):
    def setUp(self):
        self._orig_threshold = pycauset.get_io_streaming_threshold()
        pycauset.set_io_streaming_threshold(16)
        pycauset.clear_io_traces()

    def tearDown(self):
        pycauset.set_io_streaming_threshold(self._orig_threshold)
        pycauset.clear_io_traces()

    def _small_matrix(self):
        return pycauset.matrix([[1.0, 2.0], [3.0, 4.0]])

    def _file_backed_fake(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(b"0123456789")
        tmp.flush()
        tmp.close()
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.remove(tmp.name))

        class FakeMatrix:
            def __init__(self, path):
                self.shape = (2, 2)
                self._path = path

            def get_backing_file(self):
                return self._path

            def get(self, i, j):
                return float(i + j + 1)

        return FakeMatrix(tmp.name)

    def test_matmul_records_streaming_trace(self):
        a = self._small_matrix()
        b = self._small_matrix()

        _ = pycauset.matmul(a, b)

        trace = pycauset.last_io_trace()
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get("op"), "matmul")
        self.assertEqual(trace.get("route"), "streaming")
        self.assertIsNotNone(trace.get("tile_shape"))
        self.assertGreaterEqual(trace.get("queue_depth", 0), 0)
        self.assertTrue(any(op.get("estimated_bytes") for op in trace.get("operands", [])))
        self.assertTrue(str(trace.get("trace_tag", "")).startswith("matmul"))

    def test_eigvalsh_trace_is_accessible_per_op(self):
        a = pycauset.matrix([[1.0, 0.0], [0.0, 2.0]])

        _ = pycauset.eigvalsh(a)

        latest = pycauset.last_io_trace()
        eig_trace = pycauset.last_io_trace("eigvalsh")

        self.assertIsNotNone(eig_trace)
        self.assertEqual(eig_trace.get("op"), "eigvalsh")
        self.assertEqual(latest.get("trace_tag"), eig_trace.get("trace_tag"))
        self.assertIn("operands", eig_trace)
        self.assertTrue(eig_trace.get("route") in ("streaming", "direct"))

    def test_file_backed_operand_forces_streaming(self):
        a = self._file_backed_fake()
        b = self._file_backed_fake()

        _ = pycauset.matmul(a, b)

        trace = pycauset.last_io_trace()
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get("route"), "streaming")
        self.assertEqual(trace.get("reason"), "file-backed operand")
        self.assertTrue(any(op.get("backing_file") for op in trace.get("operands", [])))

    def test_threshold_none_prefers_direct_route(self):
        pycauset.set_io_streaming_threshold(None)
        pycauset.clear_io_traces()

        _ = pycauset.matmul(self._small_matrix(), self._small_matrix())

        trace = pycauset.last_io_trace()
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get("route"), "direct")
        self.assertEqual(trace.get("reason"), "no threshold configured")

    def test_eigvals_arnoldi_traced_topk(self):
        native = getattr(pycauset, "_native", None)
        have_native = native is not None and callable(getattr(native, "eigvals_arnoldi", None))

        pycauset.set_io_streaming_threshold(1)
        pycauset.clear_io_traces()

        a = pycauset.matrix([[5.0, 0.0], [0.0, 1.0]])
        _ = pycauset.eigvals_arnoldi(a, 1, 4, 1e-6)

        trace = pycauset.last_io_trace("eigvals_arnoldi")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get("op"), "eigvals_arnoldi")
        self.assertTrue(trace.get("route") in ("streaming", "direct"))
        # When native path exists, ensure we at least captured operands and a trace tag.
        self.assertTrue(any(op.get("estimated_bytes") for op in trace.get("operands", [])))
        self.assertTrue(str(trace.get("trace_tag", "")).startswith("eigvals_arnoldi"))

    def test_large_native_matrix_forces_streaming_by_threshold(self):
        # Create a modestly sized native matrix whose estimated bytes exceed a tiny threshold.
        pycauset.set_io_streaming_threshold(64)  # bytes
        pycauset.clear_io_traces()

        m = pycauset.FloatMatrix(32)  # 32x32 ~ 8 KB with float64
        # Touch a single element to avoid unused warnings; rest can stay zero.
        m.set(0, 0, 1.0)

        _ = pycauset.matmul(m, m)

        trace = pycauset.last_io_trace("matmul")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get("route"), "streaming")
        self.assertEqual(trace.get("reason"), "estimated bytes exceed threshold")
        self.assertTrue(any(op.get("estimated_bytes") for op in trace.get("operands", [])))
        self.assertEqual(trace.get("impl"), "streaming_python")

    def test_invert_forces_streaming_path(self):
        pycauset.set_io_streaming_threshold(64)
        pycauset.clear_io_traces()

        m = pycauset.FloatMatrix(8)
        for i in range(8):
            m.set(i, i, 2.0)

        _ = pycauset.invert(m)

        trace = pycauset.last_io_trace("invert")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get("route"), "streaming")
        self.assertEqual(trace.get("impl"), "streaming_python")

    def test_eigvalsh_forces_streaming_path(self):
        pycauset.set_io_streaming_threshold(64)
        pycauset.clear_io_traces()

        m = pycauset.FloatMatrix(8)
        for i in range(8):
            m.set(i, i, float(i + 1))

        _ = pycauset.eigvalsh(m)

        trace = pycauset.last_io_trace("eigvalsh")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get("route"), "streaming")
        self.assertEqual(trace.get("impl"), "streaming_python")

    def test_eigvals_arnoldi_forces_streaming_path(self):
        pycauset.set_io_streaming_threshold(64)
        pycauset.clear_io_traces()

        m = pycauset.FloatMatrix(8)
        for i in range(8):
            m.set(i, i, float(10 - i))

        _ = pycauset.eigvals_arnoldi(m, 2, 4, 1e-6)

        trace = pycauset.last_io_trace("eigvals_arnoldi")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get("route"), "streaming")
        self.assertEqual(trace.get("impl"), "streaming_python")


if __name__ == "__main__":
    unittest.main()
