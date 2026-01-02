import tempfile
import unittest
from pathlib import Path

import pycauset


class TempStorageLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_keep = pycauset.keep_temp_files
        self._orig_root = pycauset._storage_root()
        self._orig_threshold = pycauset.get_memory_threshold()
        self._orig_io_threshold = pycauset.get_io_streaming_threshold()
        pycauset.clear_io_traces()

    def tearDown(self) -> None:
        pycauset.keep_temp_files = self._orig_keep
        try:
            pycauset.set_memory_threshold(self._orig_threshold)
        except Exception:
            pass
        try:
            pycauset.set_io_streaming_threshold(self._orig_io_threshold)
        except Exception:
            pass
        try:
            pycauset.set_backing_dir(self._orig_root)
        except Exception:
            pass
        pycauset.clear_io_traces()

    def test_stale_temp_cleaned_on_set_backing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stale = tmp_path / "leftover.tmp"
            stale.write_text("junk")

            pycauset.set_backing_dir(tmp_path)

            self.assertFalse(stale.exists())

    def test_keep_temp_files_toggle_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pycauset.set_backing_dir(tmp_path)
            pycauset.set_memory_threshold(1)

            pycauset.keep_temp_files = False
            first_path = tmp_path / "first.tmp"
            first_path.write_text("spill")
            pycauset._storage.record_temporary_file(first_path)

            self.assertTrue(first_path.exists())

            pycauset._runtime.cleanup_all_roots(keep_temp_files=pycauset.keep_temp_files)
            self.assertFalse(first_path.exists())

            pycauset.keep_temp_files = True
            second_path = tmp_path / "second.tmp"
            second_path.write_text("spill")
            pycauset._storage.record_temporary_file(second_path)

            pycauset._runtime.cleanup_all_roots(keep_temp_files=pycauset.keep_temp_files)
            self.assertTrue(second_path.exists())
            second_path.unlink(missing_ok=True)

    def test_io_trace_reports_spill_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pycauset.set_backing_dir(tmp_path)
            pycauset.set_memory_threshold(1)
            pycauset.set_io_streaming_threshold(16)
            pycauset.clear_io_traces()

            a = pycauset.TriangularBitMatrix(8)
            b = pycauset.TriangularBitMatrix(8)
            out = pycauset.matmul(a, b)

            trace = pycauset.last_io_trace("matmul")
            self.assertIsNotNone(trace)
            storage = trace.get("storage", {})
            self.assertTrue(storage.get("spilled"))
            self.assertTrue(storage.get("temporary_files"))
            events = trace.get("events", [])
            self.assertTrue(any(evt.get("type") == "io" for evt in events))
            self.assertTrue(any(evt.get("type") == "compute" for evt in events))

            for obj in (a, b, out):
                try:
                    obj.close()
                except Exception:
                    pass

            pycauset._runtime.cleanup_all_roots(keep_temp_files=False)


if __name__ == "__main__":
    unittest.main()
