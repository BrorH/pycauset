import unittest
import shutil
from pathlib import Path
import uuid
import struct

import pycauset


class TestPropertiesPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_properties_persistence_output")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        import gc

        gc.collect()

        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                print(f"WARNING: Could not clean up {self.test_dir} due to file locking.")

    def test_gospel_properties_tristate_roundtrip(self):
        m = None
        m2 = None
        try:
            m = pycauset.FloatMatrix(2)

            # Explicit False must persist as False (distinct from missing/unset).
            m.properties = {
                "is_identity": False,
                # None means "unset"; setter drops it and it should not round-trip.
                "is_zero": None,
            }

            path = self.test_dir / "gospel_props.pycauset"
            pycauset.save(m, path)

            m2 = pycauset.load(path)
            self.assertIn("is_identity", m2.properties)
            self.assertIs(m2.properties.get("is_identity"), False)
            self.assertNotIn("is_zero", m2.properties)

            # Unset means missing.
            m.properties = {}
            path2 = self.test_dir / "gospel_props_unset.pycauset"
            pycauset.save(m, path2)
            m2.close()
            m2 = pycauset.load(path2)
            self.assertNotIn("is_identity", m2.properties)
        finally:
            if m2 is not None:
                m2.close()
            if m is not None:
                m.close()

    def _tamper_payload_uuid_only(self, path: Path) -> None:
        # Internal helper for signature-mismatch tests: change top-level payload_uuid
        # without updating cached.* signatures, keeping metadata block length stable.
        import pycauset._internal.persistence as persistence

        active, _, _ = persistence._read_active_slot_and_typed_metadata(path)  # type: ignore[attr-defined]
        meta_off = int(active["metadata_offset"])
        meta_len = int(active["metadata_length"])

        with path.open("r+b") as f:
            f.seek(meta_off)
            block = f.read(meta_len)
            if len(block) != meta_len:
                raise AssertionError("failed to read metadata block")

            if block[:4] != persistence._METADATA_BLOCK_MAGIC:  # type: ignore[attr-defined]
                raise AssertionError("metadata block magic mismatch")

            payload_len = struct.unpack("<Q", block[16:24])[0]
            old_payload = block[32:]
            if len(old_payload) != payload_len:
                raise AssertionError("metadata payload length mismatch")

            typed = persistence._decode_metadata_top_map(old_payload)  # type: ignore[attr-defined]
            old_uuid = typed.get("payload_uuid")
            if not isinstance(old_uuid, str):
                raise AssertionError("missing payload_uuid")

            # Preserve the on-disk UUID string format to keep payload size stable.
            # (Some builds use UUID hex without dashes.)
            if len(old_uuid) == 32:
                new_uuid = uuid.uuid4().hex
            else:
                new_uuid = str(uuid.uuid4())
            if len(new_uuid) != len(old_uuid):
                raise AssertionError("expected stable uuid string length")

            typed["payload_uuid"] = new_uuid
            new_payload = persistence._encode_metadata_top_map(typed)  # type: ignore[attr-defined]
            if len(new_payload) != len(old_payload):
                raise AssertionError("expected metadata payload size to remain stable")

            new_crc = persistence._crc32(new_payload)  # type: ignore[attr-defined]
            new_header = block[:24] + struct.pack("<I", new_crc) + block[28:32]

            f.seek(meta_off)
            f.write(new_header)
            f.write(new_payload)

    def test_cached_scalars_surface_only_with_valid_signature(self):
        m = None
        m2 = None
        try:
            m = pycauset.FloatMatrix(3)
            m.set(0, 0, 2.0)
            m.set(1, 1, 3.0)
            m.set(2, 2, 4.0)

            # Populate cached-derived values into runtime properties.
            self.assertEqual(m.trace(), 9.0)
            self.assertEqual(m.determinant(), 24.0)
            self.assertAlmostEqual(pycauset.norm(m), 5.385164807134504)
            self.assertEqual(pycauset.sum(m), 9.0 + 0.0j)
            self.assertIn("trace", m.properties)
            self.assertIn("determinant", m.properties)
            self.assertIn("norm", m.properties)
            self.assertIn("sum", m.properties)

            path = self.test_dir / "cached_scalars.pycauset"
            pycauset.save(m, path)

            # Break validity by changing payload_uuid only.
            self._tamper_payload_uuid_only(path)

            m2 = pycauset.load(path)

            # Cached-derived scalars should NOT be surfaced when signature mismatches.
            self.assertNotIn("trace", m2.properties)
            self.assertNotIn("determinant", m2.properties)
            self.assertNotIn("norm", m2.properties)
            self.assertNotIn("sum", m2.properties)

            # Calls still work; they just compute and then populate properties.
            self.assertEqual(m2.trace(), 9.0)
            self.assertEqual(m2.determinant(), 24.0)
            self.assertAlmostEqual(pycauset.norm(m2), 5.385164807134504)
            self.assertEqual(pycauset.sum(m2), 9.0 + 0.0j)
            self.assertEqual(m2.properties.get("trace"), 9.0)
            self.assertEqual(m2.properties.get("determinant"), 24.0)
            self.assertAlmostEqual(m2.properties.get("norm"), 5.385164807134504)
            self.assertEqual(m2.properties.get("sum"), 9.0 + 0.0j)
        finally:
            if m2 is not None:
                m2.close()
            if m is not None:
                m.close()


if __name__ == "__main__":
    unittest.main()
