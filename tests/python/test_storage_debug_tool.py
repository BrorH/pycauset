import shutil
import unittest
from pathlib import Path

import pycauset

from pycauset._internal import storage_debug


class TestStorageDebugTool(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_storage_debug_tool_tmp")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_summarize_container_on_known_good_file(self):
        m = pycauset.IntegerMatrix(8)
        path = self.test_dir / "ok.pycauset"
        try:
            pycauset.save(m, path)

            info = storage_debug.summarize_container(path)
            self.assertEqual(Path(info["path"]), path)
            self.assertEqual(info["preamble"]["format_version"], 1)
            self.assertEqual(info["active_slot"], "A")

            slot_a = info["slot_a"]
            self.assertTrue(slot_a.valid)
            self.assertTrue(slot_a.crc_ok)
            self.assertEqual(slot_a.payload_offset % 4096, 0)
            self.assertEqual(slot_a.metadata_offset % 16, 0)

            # slot B may be valid (same generation) or empty/invalid depending on writer.
            self.assertIn(info["active_slot"], ("A", "B"))
        finally:
            m.close()


if __name__ == "__main__":
    unittest.main()
