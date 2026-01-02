import numpy as np
import pycauset

from pycauset._internal import persistence as _persistence


def test_large_metadata_update_does_not_move_payload(tmp_path):
    n = 128
    arr = np.arange(n * n, dtype=np.float64).reshape(n, n)
    m = pycauset.matrix(arr)
    path = tmp_path / "large_meta.pycauset"

    try:
        pycauset.save(m, path)
        slot_before, _, meta_before = _persistence._read_active_slot_and_typed_metadata(path)
        with path.open("rb") as f:
            f.seek(int(slot_before["payload_offset"]))
            payload_before = f.read(int(slot_before["payload_length"]))

        # Metadata-only tweak
        m.properties["phase"] = "2"
        pycauset.save(m, path)

        slot_after, _, meta_after = _persistence._read_active_slot_and_typed_metadata(path)
        with path.open("rb") as f:
            f.seek(int(slot_after["payload_offset"]))
            payload_after = f.read(int(slot_after["payload_length"]))

        assert slot_before["payload_offset"] == slot_after["payload_offset"]
        assert slot_before["payload_length"] == slot_after["payload_length"]
        assert payload_before == payload_after
        assert meta_after.get("properties", {}).get("phase") == "2"
        assert meta_before.get("payload_uuid") != meta_after.get("payload_uuid")
    finally:
        m.close()
