from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import persistence as _persistence


@dataclass(frozen=True)
class SlotSummary:
    name: str
    crc_ok: bool
    generation: int
    payload_offset: int
    payload_length: int
    metadata_offset: int
    metadata_length: int
    hot_offset: int
    hot_length: int
    valid: bool


def _slot_valid(slot: dict[str, int], *, file_size: int) -> bool:
    if not slot.get("crc_ok"):
        return False
    if slot["payload_offset"] % 4096 != 0:
        return False
    if slot["metadata_offset"] % 16 != 0:
        return False
    if slot["payload_offset"] + slot["payload_length"] > file_size:
        return False
    if slot["metadata_offset"] + slot["metadata_length"] > file_size:
        return False
    return True


def summarize_container(path: str | Path) -> dict[str, Any]:
    """Return a best-effort summary of a `.pycauset` container header.

    This is an internal debugging helper intended for tests and developer tooling.
    It does not scan payload or search for alternative metadata blocks.
    """
    path = Path(path)
    file_size = path.stat().st_size

    with path.open("rb") as f:
        preamble = f.read(16)
        if len(preamble) != 16:
            raise ValueError("invalid file: truncated header")

        magic, version, endian, header_bytes, reserved = _persistence.struct.unpack("<8sIBHB", preamble)
        f.seek(_persistence._SLOT_A_OFFSET)
        slot_a = _persistence._unpack_slot(f.read(_persistence._SLOT_SIZE))
        f.seek(_persistence._SLOT_B_OFFSET)
        slot_b = _persistence._unpack_slot(f.read(_persistence._SLOT_SIZE))

    slot_a_valid = _slot_valid(slot_a, file_size=file_size)
    slot_b_valid = _slot_valid(slot_b, file_size=file_size)

    def _mk(name: str, slot: dict[str, int], valid: bool) -> SlotSummary:
        return SlotSummary(
            name=name,
            crc_ok=bool(slot.get("crc_ok")),
            generation=int(slot.get("generation", 0)),
            payload_offset=int(slot.get("payload_offset", 0)),
            payload_length=int(slot.get("payload_length", 0)),
            metadata_offset=int(slot.get("metadata_offset", 0)),
            metadata_length=int(slot.get("metadata_length", 0)),
            hot_offset=int(slot.get("hot_offset", 0)),
            hot_length=int(slot.get("hot_length", 0)),
            valid=valid,
        )

    active = None
    if slot_a_valid or slot_b_valid:
        candidates = []
        if slot_a_valid:
            candidates.append(("A", slot_a))
        if slot_b_valid:
            candidates.append(("B", slot_b))
        active = max(candidates, key=lambda kv: int(kv[1]["generation"]))[0]

    return {
        "path": str(path),
        "file_size": int(file_size),
        "preamble": {
            "magic": magic,
            "format_version": int(version),
            "endian": int(endian),
            "header_bytes": int(header_bytes),
            "reserved": int(reserved),
        },
        "slot_a": _mk("A", slot_a, slot_a_valid),
        "slot_b": _mk("B", slot_b, slot_b_valid),
        "active_slot": active,
    }
