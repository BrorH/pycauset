from __future__ import annotations

import os
import struct
import shutil
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Protocol

import uuid


def _compute_view_signature(*, is_transposed: bool, is_conjugated: bool, scalar: object) -> str:
    if isinstance(scalar, dict) and "real" in scalar and "imag" in scalar:
        scalar = complex(float(scalar["real"]), float(scalar["imag"]))
    if isinstance(scalar, complex):
        sr = float(scalar.real)
        si = float(scalar.imag)
    else:
        sr = float(scalar)  # type: ignore[arg-type]
        si = 0.0

    return f"t={int(bool(is_transposed))};c={int(bool(is_conjugated))};sr={sr:g};si={si:g}"


class _HasCopyStorage(Protocol):
    def copy_storage(self, path: str) -> None: ...


_MAGIC = b"PYCAUSET"  # 8 bytes
_FORMAT_VERSION = 1
_ENDIAN_LITTLE = 1
_HEADER_BYTES = 4096
_RAW_STORAGE_HEADER_BYTES = 64
_SLOT_SIZE = 128
_SLOT_A_OFFSET = 16
_SLOT_B_OFFSET = _SLOT_A_OFFSET + _SLOT_SIZE
_METADATA_BLOCK_MAGIC = b"PCMB"
_METADATA_BLOCK_VERSION = 1
_METADATA_ENCODING_VERSION = 1

_CACHED_DERIVED_PROPERTY_KEYS = {"trace", "determinant", "norm", "sum", "eigenvalues"}


def _extract_properties_for_persistence(obj: Any) -> dict[str, Any]:
    props = getattr(obj, "properties", None)
    if not isinstance(props, Mapping):
        return {}
    out: dict[str, Any] = {}
    for k, v in props.items():
        if not isinstance(k, str):
            continue
        if v is None:
            continue
        # Keep only values we can safely encode deterministically.
        # Note: some cached-derived values may be complex; they are serialized in
        # a stable representation under `cached.*` by the save routine.
        if isinstance(v, (bool, int, float, complex, str, bytes, bytearray, list, tuple, dict)):
            out[k] = v
    return out


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError("alignment must be a power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def _raw_storage_header_bytes(path: Path) -> int:
    """Return header size for raw storage files created by MemoryMapper.

    copy_storage writes a 64-byte header (magic/version + zeroed reserved bytes)
    followed by raw payload bytes. Detect that header so we can persist only
    the data payload in the container.
    """

    try:
        with path.open("rb") as f:
            header = f.read(_RAW_STORAGE_HEADER_BYTES)
    except OSError:
        return 0

    if len(header) != _RAW_STORAGE_HEADER_BYTES:
        return 0
    if header[:8] != _MAGIC:
        return 0

    version = struct.unpack("<I", header[8:12])[0]
    if version != 1:
        return 0

    # MemoryMapper uses a "simple" header (reserved bytes all zero).
    if any(b != 0 for b in header[12:]):
        return 0

    return _RAW_STORAGE_HEADER_BYTES


class _Cursor:
    def __init__(self, data: bytes):
        self._data = data
        self._i = 0

    def remaining(self) -> int:
        return len(self._data) - self._i

    def take(self, n: int) -> bytes:
        if n < 0 or self._i + n > len(self._data):
            raise ValueError("truncated metadata payload")
        out = self._data[self._i : self._i + n]
        self._i += n
        return out

    def u8(self) -> int:
        return struct.unpack("<B", self.take(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8))[0]


def _encode_key(key: str) -> bytes:
    if not isinstance(key, str):
        raise TypeError("metadata map keys must be strings")
    raw = key.encode("utf-8")
    if len(raw) > 0xFFFF:
        raise ValueError("metadata key too long")
    return struct.pack("<H", len(raw)) + raw


def _encode_value(value: Any) -> bytes:
    # Tags (v1):
    # 0x01 Bool, 0x02 I64, 0x03 U64, 0x04 F64, 0x05 String, 0x06 Bytes, 0x07 Array, 0x08 Map
    if isinstance(value, bool):
        return b"\x01" + struct.pack("<B", 1 if value else 0)

    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            return b"\x02" + struct.pack("<q", value)
        return b"\x03" + struct.pack("<Q", value)

    if isinstance(value, float):
        return b"\x04" + struct.pack("<d", value)

    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"\x05" + struct.pack("<I", len(raw)) + raw

    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        return b"\x06" + struct.pack("<I", len(raw)) + raw

    if isinstance(value, (list, tuple)):
        out = bytearray()
        out += b"\x07" + struct.pack("<I", len(value))
        for item in value:
            out += _encode_value(item)
        return bytes(out)

    if isinstance(value, dict):
        out = bytearray()
        keys = list(value.keys())
        for k in keys:
            if not isinstance(k, str):
                raise TypeError("metadata map keys must be strings")

        keys.sort()
        out += b"\x08" + struct.pack("<I", len(keys))
        for k in keys:
            out += _encode_key(k)
            out += _encode_value(value[k])
        return bytes(out)

    raise TypeError(f"unsupported metadata type: {type(value)!r}")


def _decode_value(cur: _Cursor) -> Any:
    tag = cur.u8()

    if tag == 0x01:
        return bool(cur.u8())
    if tag == 0x02:
        return cur.i64()
    if tag == 0x03:
        return cur.u64()
    if tag == 0x04:
        return cur.f64()
    if tag == 0x05:
        n = cur.u32()
        return cur.take(n).decode("utf-8")
    if tag == 0x06:
        n = cur.u32()
        return cur.take(n)
    if tag == 0x07:
        count = cur.u32()
        return [_decode_value(cur) for _ in range(count)]
    if tag == 0x08:
        count = cur.u32()
        out: dict[str, Any] = {}
        for _ in range(count):
            key_len = cur.u16()
            key = cur.take(key_len).decode("utf-8")
            out[key] = _decode_value(cur)
        return out

    raise ValueError(f"unknown metadata tag: 0x{tag:02x}")


def _encode_metadata_top_map(meta: dict[str, Any]) -> bytes:
    if not isinstance(meta, dict):
        raise TypeError("metadata must be a dict")
    return _encode_value(meta)


def _decode_metadata_top_map(payload: bytes) -> dict[str, Any]:
    cur = _Cursor(payload)
    value = _decode_value(cur)
    if cur.remaining() != 0:
        raise ValueError("trailing bytes in metadata payload")
    if not isinstance(value, dict):
        raise ValueError("top-level metadata must be a map")
    return value


def _pack_preamble() -> bytes:
    # 16 bytes total: 8 + 4 + 1 + 2 + 1
    return struct.pack(
        "<8sIBHB",
        _MAGIC,
        _FORMAT_VERSION,
        _ENDIAN_LITTLE,
        _HEADER_BYTES,
        0,
    )


def _pack_slot(
    *,
    generation: int,
    payload_offset: int,
    payload_length: int,
    metadata_offset: int,
    metadata_length: int,
    hot_offset: int,
    hot_length: int,
) -> bytes:
    head = struct.pack(
        "<QQQQQQQ",
        generation,
        payload_offset,
        payload_length,
        metadata_offset,
        metadata_length,
        hot_offset,
        hot_length,
    )
    crc = _crc32(head)
    tail = struct.pack("<I", crc) + (b"\x00" * 68)
    slot = head + tail
    if len(slot) != _SLOT_SIZE:
        raise AssertionError("slot size mismatch")
    return slot


def _unpack_slot(raw: bytes) -> dict[str, int]:
    if len(raw) != _SLOT_SIZE:
        raise ValueError("invalid slot length")
    fields = struct.unpack("<QQQQQQQ", raw[:56])
    expected_crc = struct.unpack("<I", raw[56:60])[0]
    actual_crc = _crc32(raw[:56])
    return {
        "generation": int(fields[0]),
        "payload_offset": int(fields[1]),
        "payload_length": int(fields[2]),
        "metadata_offset": int(fields[3]),
        "metadata_length": int(fields[4]),
        "hot_offset": int(fields[5]),
        "hot_length": int(fields[6]),
        "crc_ok": 1 if expected_crc == actual_crc else 0,
    }


def _is_new_container(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(len(_MAGIC)) == _MAGIC
    except OSError:
        return False


def _read_active_slot_and_typed_metadata(path: Path) -> tuple[dict[str, int], str, dict[str, Any]]:
    """Return (active_slot, active_slot_name, typed_metadata_map)."""
    file_size = path.stat().st_size
    with path.open("rb") as f:
        preamble = f.read(16)
        if len(preamble) != 16:
            raise ValueError("invalid file: truncated header")

        magic, version, endian, header_bytes, reserved = struct.unpack("<8sIBHB", preamble)
        if magic != _MAGIC:
            raise ValueError("invalid file: magic mismatch")
        if version != _FORMAT_VERSION:
            raise ValueError(f"unsupported format version: {version}")
        if endian != _ENDIAN_LITTLE:
            raise ValueError("unsupported endian marker")
        if header_bytes != _HEADER_BYTES:
            raise ValueError(f"unsupported header size: {header_bytes}")
        if reserved != 0:
            raise ValueError("invalid file: reserved header byte is non-zero")

        f.seek(_SLOT_A_OFFSET)
        slot_a = _unpack_slot(f.read(_SLOT_SIZE))
        f.seek(_SLOT_B_OFFSET)
        slot_b = _unpack_slot(f.read(_SLOT_SIZE))

        def _slot_valid(s: dict[str, int]) -> bool:
            if not s.get("crc_ok"):
                return False
            if s["payload_offset"] % 4096 != 0:
                return False
            if s["metadata_offset"] % 16 != 0:
                return False
            if s["payload_offset"] + s["payload_length"] > file_size:
                return False
            if s["metadata_offset"] + s["metadata_length"] > file_size:
                return False
            return True

        candidates: list[tuple[str, dict[str, int]]] = []
        if _slot_valid(slot_a):
            candidates.append(("A", slot_a))
        if _slot_valid(slot_b):
            candidates.append(("B", slot_b))
        if not candidates:
            raise ValueError("invalid file: no valid header slot")

        # Deterministic tie-break: generation desc, then slot name.
        active_name, active = max(candidates, key=lambda it: (it[1]["generation"], it[0]))

        f.seek(active["metadata_offset"])
        block = f.read(active["metadata_length"])
        if len(block) != active["metadata_length"]:
            raise ValueError("invalid file: truncated metadata block")

    if len(block) < 32:
        raise ValueError("invalid file: metadata block too small")
    if block[:4] != _METADATA_BLOCK_MAGIC:
        raise ValueError("invalid file: metadata block magic mismatch")

    block_version, encoding_version, reserved0 = struct.unpack("<III", block[4:16])
    if reserved0 != 0:
        raise ValueError("invalid file: metadata reserved field non-zero")
    if block_version != _METADATA_BLOCK_VERSION:
        raise ValueError(f"unsupported metadata block version: {block_version}")
    if encoding_version != _METADATA_ENCODING_VERSION:
        raise ValueError(f"unsupported metadata encoding version: {encoding_version}")

    payload_length = struct.unpack("<Q", block[16:24])[0]
    payload_crc32 = struct.unpack("<I", block[24:28])[0]
    reserved1 = struct.unpack("<I", block[28:32])[0]
    if reserved1 != 0:
        raise ValueError("invalid file: metadata reserved2 field non-zero")

    expected_total = 32 + payload_length
    if expected_total != len(block):
        raise ValueError("invalid file: metadata block length mismatch")

    payload = block[32:]
    if _crc32(payload) != payload_crc32:
        raise ValueError("invalid file: metadata payload checksum failed")

    typed_meta = _decode_metadata_top_map(payload)
    return active, active_name, typed_meta


def read_typed_metadata(path: str | Path) -> dict[str, Any]:
    """Read and return the top-level typed metadata map from a `.pycauset` container."""
    path = Path(path)
    if not _is_new_container(path):
        raise ValueError("unsupported file format (expected new .pycauset container)")
    _, _, typed_meta = _read_active_slot_and_typed_metadata(path)
    return typed_meta


def try_get_cached_big_blob_ref(
    path: str | Path,
    *,
    name: str,
    view_signature: str | None = None,
) -> dict[str, Any] | None:
    """Return the cached big-blob reference for cached.<name> if present and valid.

    Expected shape (typed metadata):

    cached[name] = {
      "value": {"ref_kind": "sibling_object_store", "object_id": "..."},
      "signature": {"payload_uuid": "...", "view_signature": "..."}
    }
    """
    typed_meta = read_typed_metadata(path)

    payload_uuid = typed_meta.get("payload_uuid")
    if not isinstance(payload_uuid, str) or not payload_uuid:
        raise ValueError("invalid file: missing payload_uuid in typed metadata")

    cached = typed_meta.get("cached")
    if not isinstance(cached, dict):
        return None
    entry = cached.get(name)
    if not isinstance(entry, dict):
        return None

    sig = entry.get("signature")
    if not (isinstance(sig, dict) and sig.get("payload_uuid") == payload_uuid):
        return None

    if view_signature is not None and sig.get("view_signature") != view_signature:
        return None

    value = entry.get("value")
    if not isinstance(value, dict):
        return None
    ref_kind = value.get("ref_kind")
    object_id = value.get("object_id")
    if not (isinstance(ref_kind, str) and isinstance(object_id, str) and object_id):
        return None
    return {"ref_kind": ref_kind, "object_id": object_id}


def _object_store_dir_for_base(base_path: str | Path) -> Path:
    base_path = Path(base_path)
    # Sidecar directory living next to the base snapshot. Keeps caches colocated.
    return base_path.with_suffix(base_path.suffix + ".objects")


def object_store_path_for_id(base_path: str | Path, *, object_id: str) -> Path:
    store_dir = _object_store_dir_for_base(base_path)
    return store_dir / f"{object_id}.pycauset"


def new_object_id() -> str:
    """Return a stable identifier for a persisted big-blob object."""
    return uuid.uuid4().hex


def write_cached_big_blob_ref(
    base_path: str | Path,
    *,
    name: str,
    ref_kind: str,
    object_id: str,
    signature: dict[str, Any],
) -> None:
    """Append a metadata update that stores a big-blob cache reference under cached.<name>.

    This does not write the big-blob object itself. The caller must ensure the referenced
    object is fully written/published before calling this.
    """
    base_path = Path(base_path)
    if not _is_new_container(base_path):
        raise ValueError("unsupported file format (expected new .pycauset container)")

    active, active_name, typed_meta = _read_active_slot_and_typed_metadata(base_path)

    inactive_slot_offset = _SLOT_B_OFFSET if active_name == "A" else _SLOT_A_OFFSET
    new_generation = int(active.get("generation", 0)) + 1

    cached = typed_meta.get("cached")
    if not isinstance(cached, dict):
        cached = {}

    cached[name] = {
        "value": {"ref_kind": str(ref_kind), "object_id": str(object_id)},
        "signature": dict(signature),
    }
    typed_meta["cached"] = cached

    with base_path.open("r+b") as f:
        f.seek(0, os.SEEK_END)
        cur = f.tell()
        meta_offset = _align_up(cur, 16)
        if meta_offset != cur:
            f.write(b"\x00" * (meta_offset - cur))

        meta_payload = _encode_metadata_top_map(typed_meta)
        block_payload_crc = _crc32(meta_payload)
        block_header = (
            _METADATA_BLOCK_MAGIC
            + struct.pack(
                "<III",
                _METADATA_BLOCK_VERSION,
                _METADATA_ENCODING_VERSION,
                0,
            )
            + struct.pack("<Q", len(meta_payload))
            + struct.pack("<I", block_payload_crc)
            + struct.pack("<I", 0)
        )
        if len(block_header) != 32:
            raise AssertionError("metadata block header size mismatch")
        f.write(block_header)
        f.write(meta_payload)
        meta_length = 32 + len(meta_payload)

        new_slot = _pack_slot(
            generation=new_generation,
            payload_offset=int(active["payload_offset"]),
            payload_length=int(active["payload_length"]),
            metadata_offset=int(meta_offset),
            metadata_length=int(meta_length),
            hot_offset=int(active.get("hot_offset", 0)),
            hot_length=int(active.get("hot_length", 0)),
        )
        f.seek(inactive_slot_offset)
        f.write(new_slot)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def _read_new_container_metadata_and_offset(path: Path) -> tuple[dict[str, Any], int]:
    file_size = path.stat().st_size
    with path.open("rb") as f:
        preamble = f.read(16)
        if len(preamble) != 16:
            raise ValueError("invalid file: truncated header")

        magic, version, endian, header_bytes, reserved = struct.unpack("<8sIBHB", preamble)
        if magic != _MAGIC:
            raise ValueError("invalid file: magic mismatch")
        if version != _FORMAT_VERSION:
            raise ValueError(f"unsupported format version: {version}")
        if endian != _ENDIAN_LITTLE:
            raise ValueError("unsupported endian marker")
        if header_bytes != _HEADER_BYTES:
            raise ValueError(f"unsupported header size: {header_bytes}")
        if reserved != 0:
            raise ValueError("invalid file: reserved header byte is non-zero")

        f.seek(_SLOT_A_OFFSET)
        slot_a = _unpack_slot(f.read(_SLOT_SIZE))
        f.seek(_SLOT_B_OFFSET)
        slot_b = _unpack_slot(f.read(_SLOT_SIZE))

        def _slot_valid(s: dict[str, int]) -> bool:
            if not s.get("crc_ok"):
                return False
            if s["payload_offset"] % 4096 != 0:
                return False
            if s["metadata_offset"] % 16 != 0:
                return False
            if s["payload_offset"] + s["payload_length"] > file_size:
                return False
            if s["metadata_offset"] + s["metadata_length"] > file_size:
                return False
            return True

        candidates = [s for s in (slot_a, slot_b) if _slot_valid(s)]
        if not candidates:
            raise ValueError("invalid file: no valid header slot")

        active = max(candidates, key=lambda s: s["generation"])

        f.seek(active["metadata_offset"])
        block = f.read(active["metadata_length"])
        if len(block) != active["metadata_length"]:
            raise ValueError("invalid file: truncated metadata block")

    if len(block) < 32:
        raise ValueError("invalid file: metadata block too small")

    block_magic = block[:4]
    if block_magic != _METADATA_BLOCK_MAGIC:
        raise ValueError("invalid file: metadata block magic mismatch")

    block_version, encoding_version, reserved0 = struct.unpack("<III", block[4:16])
    if reserved0 != 0:
        raise ValueError("invalid file: metadata reserved field non-zero")
    if block_version != _METADATA_BLOCK_VERSION:
        raise ValueError(f"unsupported metadata block version: {block_version}")
    if encoding_version != _METADATA_ENCODING_VERSION:
        raise ValueError(f"unsupported metadata encoding version: {encoding_version}")

    payload_length = struct.unpack("<Q", block[16:24])[0]
    payload_crc32 = struct.unpack("<I", block[24:28])[0]
    reserved1 = struct.unpack("<I", block[28:32])[0]
    if reserved1 != 0:
        raise ValueError("invalid file: metadata reserved2 field non-zero")

    expected_total = 32 + payload_length
    if expected_total != len(block):
        raise ValueError("invalid file: metadata block length mismatch")

    payload = block[32:]
    if _crc32(payload) != payload_crc32:
        raise ValueError("invalid file: metadata payload checksum failed")

    typed_meta = _decode_metadata_top_map(payload)
    if not isinstance(typed_meta, dict):
        raise ValueError("invalid file: metadata top-level is not a map")

    # payload begins at payload_offset
    return typed_meta, int(active["payload_offset"])


class PersistenceDeps(Protocol):
    # Types
    CausalSet: type

    TriangularBitMatrix: type | None
    DenseBitMatrix: type | None
    FloatMatrix: type | None
    Float16Matrix: type | None
    Float32Matrix: type | None
    ComplexFloat16Matrix: type | None
    ComplexFloat32Matrix: type | None
    ComplexFloat64Matrix: type | None
    IntegerMatrix: type | None
    Int8Matrix: type | None
    Int16Matrix: type | None
    Int64Matrix: type | None
    UInt8Matrix: type | None
    UInt16Matrix: type | None
    UInt32Matrix: type | None
    UInt64Matrix: type | None
    TriangularFloatMatrix: type | None
    TriangularIntegerMatrix: type | None

    FloatVector: type | None
    Float16Vector: type | None
    Float32Vector: type | None
    ComplexFloat16Vector: type | None
    ComplexFloat32Vector: type | None
    ComplexFloat64Vector: type | None
    Int8Vector: type | None
    IntegerVector: type | None
    Int16Vector: type | None
    Int64Vector: type | None
    UInt8Vector: type | None
    UInt16Vector: type | None
    UInt32Vector: type | None
    UInt64Vector: type | None
    BitVector: type | None
    UnitVector: type | None

    IdentityMatrix: type | None

    # Native module
    native: Any


def save(obj: Any, path: str | Path, *, deps: PersistenceDeps) -> None:
    """Save a matrix or vector to a file (new single-file format)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Internal BlockMatrix persistence (Phase E).
    # This is not part of the public API surface; it exists to support the
    # block-matrix roadmap without introducing a new file format.
    try:
        from .blockmatrix import BlockMatrix  # local import to avoid cycles
        from .thunks import ThunkBlock
    except Exception:  # pragma: no cover
        BlockMatrix = None  # type: ignore[assignment]
        ThunkBlock = None  # type: ignore[assignment]

    if BlockMatrix is not None and isinstance(obj, BlockMatrix):
        _save_blockmatrix(obj, path, deps=deps, BlockMatrix=BlockMatrix, ThunkBlock=ThunkBlock)
        return

    temp_raw = path.with_suffix(".raw_tmp")

    causet_obj: Any | None = None
    payload_obj = obj

    # copy_storage now writes raw data to the file
    if isinstance(obj, deps.CausalSet):
        causet_obj = obj
        payload_obj = obj.causal_matrix

    if hasattr(payload_obj, "copy_storage"):
        payload_obj.copy_storage(str(temp_raw))
    else:
        raise TypeError("Object does not support saving (missing copy_storage)")

    try:
        is_transposed = getattr(payload_obj, "is_transposed", False)
        if callable(is_transposed):
            is_transposed = is_transposed()

        is_conjugated = getattr(payload_obj, "is_conjugated", False)
        if callable(is_conjugated):
            is_conjugated = is_conjugated()

        if hasattr(payload_obj, "rows") and hasattr(payload_obj, "cols"):
            rows = payload_obj.rows() if callable(payload_obj.rows) else payload_obj.rows
            cols = payload_obj.cols() if callable(payload_obj.cols) else payload_obj.cols
        elif hasattr(payload_obj, "size"):
            rows = payload_obj.size()
            cols = 1
        else:
            rows = len(payload_obj)
            cols = 1

        scalar = getattr(payload_obj, "scalar", 1.0)
        view: dict[str, Any] = {
            "is_transposed": bool(is_transposed),
            "is_conjugated": bool(is_conjugated),
        }
        if isinstance(scalar, complex):
            view["scalar"] = {"real": float(scalar.real), "imag": float(scalar.imag)}
        else:
            view["scalar"] = float(scalar)

        meta: dict[str, Any] = {
            "rows": int(rows),
            "cols": int(cols),
            "seed": int(getattr(payload_obj, "seed", 0)),
            "payload_uuid": uuid.uuid4().hex,
            "view": view,
            "payload_layout": {"kind": "raw_native", "params": {}},
        }

        if causet_obj is not None:
            st = getattr(causet_obj, "spacetime", None)
            st_type = getattr(st, "__class__", type("_", (), {})).__name__ if st is not None else None
            st_args: dict[str, Any] = {}
            dim = getattr(st, "dimension", None)
            if callable(dim):
                try:
                    st_args["dimension"] = int(dim())
                except Exception:
                    pass

            for k in ("height", "circumference", "time_extent", "space_extent"):
                v = getattr(st, k, None)
                if v is not None:
                    try:
                        st_args[k] = float(v)
                    except Exception:
                        pass

            meta["object_type"] = "CausalSet"
            meta["n"] = int(getattr(causet_obj, "n", rows))
            meta["spacetime"] = {"type": st_type, "args": st_args}
            meta["seed"] = int(getattr(causet_obj, "_seed", meta["seed"]))

        # Determine matrix_type and data_type based on class
        if deps.TriangularBitMatrix is not None and isinstance(payload_obj, deps.TriangularBitMatrix):
            meta["matrix_type"] = "CAUSAL"
            meta["data_type"] = "BIT"
            meta["payload_layout"]["kind"] = "raw_triangular"
        elif deps.DenseBitMatrix is not None and isinstance(payload_obj, deps.DenseBitMatrix):
            meta["matrix_type"] = "DENSE_FLOAT"
            meta["data_type"] = "BIT"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Float16Matrix is not None and isinstance(payload_obj, deps.Float16Matrix):
            meta["matrix_type"] = "DENSE_FLOAT"
            meta["data_type"] = "FLOAT16"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Float32Matrix is not None and isinstance(payload_obj, deps.Float32Matrix):
            meta["matrix_type"] = "DENSE_FLOAT"
            meta["data_type"] = "FLOAT32"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.FloatMatrix is not None and isinstance(payload_obj, deps.FloatMatrix):
            meta["matrix_type"] = "DENSE_FLOAT"
            meta["data_type"] = "FLOAT64"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.ComplexFloat16Matrix is not None and isinstance(payload_obj, deps.ComplexFloat16Matrix):
            meta["matrix_type"] = "DENSE_FLOAT"
            meta["data_type"] = "COMPLEX_FLOAT16"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.ComplexFloat32Matrix is not None and isinstance(payload_obj, deps.ComplexFloat32Matrix):
            meta["matrix_type"] = "DENSE_FLOAT"
            meta["data_type"] = "COMPLEX_FLOAT32"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.ComplexFloat64Matrix is not None and isinstance(payload_obj, deps.ComplexFloat64Matrix):
            meta["matrix_type"] = "DENSE_FLOAT"
            meta["data_type"] = "COMPLEX_FLOAT64"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Int8Matrix is not None and isinstance(payload_obj, deps.Int8Matrix):
            meta["matrix_type"] = "INTEGER"
            meta["data_type"] = "INT8"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.IntegerMatrix is not None and isinstance(payload_obj, deps.IntegerMatrix):
            meta["matrix_type"] = "INTEGER"
            meta["data_type"] = "INT32"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Int16Matrix is not None and isinstance(payload_obj, deps.Int16Matrix):
            meta["matrix_type"] = "INTEGER"
            meta["data_type"] = "INT16"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Int64Matrix is not None and isinstance(payload_obj, deps.Int64Matrix):
            meta["matrix_type"] = "INTEGER"
            meta["data_type"] = "INT64"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.UInt8Matrix is not None and isinstance(payload_obj, deps.UInt8Matrix):
            meta["matrix_type"] = "INTEGER"
            meta["data_type"] = "UINT8"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.UInt16Matrix is not None and isinstance(payload_obj, deps.UInt16Matrix):
            meta["matrix_type"] = "INTEGER"
            meta["data_type"] = "UINT16"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.UInt32Matrix is not None and isinstance(payload_obj, deps.UInt32Matrix):
            meta["matrix_type"] = "INTEGER"
            meta["data_type"] = "UINT32"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.UInt64Matrix is not None and isinstance(payload_obj, deps.UInt64Matrix):
            meta["matrix_type"] = "INTEGER"
            meta["data_type"] = "UINT64"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.TriangularFloatMatrix is not None and isinstance(payload_obj, deps.TriangularFloatMatrix):
            meta["matrix_type"] = "TRIANGULAR_FLOAT"
            meta["data_type"] = "FLOAT64"
            meta["payload_layout"]["kind"] = "raw_triangular"
        elif deps.TriangularIntegerMatrix is not None and isinstance(payload_obj, deps.TriangularIntegerMatrix):
            meta["matrix_type"] = "TRIANGULAR_INTEGER"
            meta["data_type"] = "INT32"
            meta["payload_layout"]["kind"] = "raw_triangular"
        elif deps.FloatVector is not None and isinstance(payload_obj, deps.FloatVector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "FLOAT64"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Float32Vector is not None and isinstance(payload_obj, deps.Float32Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "FLOAT32"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Float16Vector is not None and isinstance(payload_obj, deps.Float16Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "FLOAT16"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.ComplexFloat16Vector is not None and isinstance(payload_obj, deps.ComplexFloat16Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "COMPLEX_FLOAT16"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.ComplexFloat32Vector is not None and isinstance(payload_obj, deps.ComplexFloat32Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "COMPLEX_FLOAT32"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.ComplexFloat64Vector is not None and isinstance(payload_obj, deps.ComplexFloat64Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "COMPLEX_FLOAT64"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Int8Vector is not None and isinstance(payload_obj, deps.Int8Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "INT8"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.IntegerVector is not None and isinstance(payload_obj, deps.IntegerVector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "INT32"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Int16Vector is not None and isinstance(payload_obj, deps.Int16Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "INT16"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.Int64Vector is not None and isinstance(payload_obj, deps.Int64Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "INT64"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.UInt8Vector is not None and isinstance(payload_obj, deps.UInt8Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "UINT8"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.UInt16Vector is not None and isinstance(payload_obj, deps.UInt16Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "UINT16"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.UInt32Vector is not None and isinstance(payload_obj, deps.UInt32Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "UINT32"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.UInt64Vector is not None and isinstance(payload_obj, deps.UInt64Vector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "UINT64"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.BitVector is not None and isinstance(payload_obj, deps.BitVector):
            meta["matrix_type"] = "VECTOR"
            meta["data_type"] = "BIT"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.UnitVector is not None and isinstance(payload_obj, deps.UnitVector):
            meta["matrix_type"] = "UNIT_VECTOR"
            meta["data_type"] = "FLOAT64"
            meta["payload_layout"]["kind"] = "raw_dense"
        elif deps.IdentityMatrix is not None and isinstance(payload_obj, deps.IdentityMatrix):
            meta["matrix_type"] = "IDENTITY"
            meta["data_type"] = "FLOAT64"
            meta["payload_layout"]["kind"] = "raw_dense"

        # Persist gospel properties (excluding cached-derived keys).
        props_all = _extract_properties_for_persistence(payload_obj)
        props_gospel = {k: v for k, v in props_all.items() if k not in _CACHED_DERIVED_PROPERTY_KEYS}
        if props_gospel:
            meta["properties"] = props_gospel

        # Persist cached-derived values in the `cached.*` shape.
        cached: dict[str, Any] = {}
        payload_uuid = meta.get("payload_uuid")
        view_sig = _compute_view_signature(
            is_transposed=bool(view["is_transposed"]),
            is_conjugated=bool(view["is_conjugated"]),
            scalar=scalar,
        )
        trace_val = props_all.get("trace")
        if trace_val is not None:
            cached["trace"] = {
                "value": float(trace_val),
                "signature": {"payload_uuid": payload_uuid, "view_signature": view_sig},
            }
        det_val = props_all.get("determinant")
        if det_val is not None:
            cached["determinant"] = {
                "value": float(det_val),
                "signature": {"payload_uuid": payload_uuid, "view_signature": view_sig},
            }
        norm_val = props_all.get("norm")
        if norm_val is not None:
            cached["norm"] = {
                "value": float(norm_val),
                "signature": {"payload_uuid": payload_uuid, "view_signature": view_sig},
            }
        sum_val = props_all.get("sum")
        if sum_val is not None:
            z = complex(sum_val)
            cached["sum"] = {
                "value": [float(z.real), float(z.imag)],
                "signature": {"payload_uuid": payload_uuid, "view_signature": view_sig},
            }
        eigen_val = props_all.get("eigenvalues")
        if eigen_val is not None:
            cached["eigenvalues"] = {
                "value": [[float(z.real), float(z.imag)] for z in eigen_val],
                "signature": {"payload_uuid": payload_uuid, "view_signature": view_sig},
            }
        if cached:
            meta["cached"] = cached

        # Write file: header placeholder, then payload, then metadata block, then header slots.
        payload_offset = _HEADER_BYTES
        raw_header = _raw_storage_header_bytes(temp_raw)
        raw_size = temp_raw.stat().st_size
        payload_length = raw_size - raw_header
        if payload_length < 0:
            raise ValueError("invalid raw payload size")

        with path.open("wb") as out_f:
            out_f.write(b"\x00" * _HEADER_BYTES)
            with temp_raw.open("rb") as in_f:
                if raw_header:
                    in_f.seek(raw_header)
                # copy_storage writes a small raw header; persist only the payload bytes.
                while True:
                    chunk = in_f.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)

            # pad to 16-byte boundary for metadata
            cur = out_f.tell()
            meta_offset = _align_up(cur, 16)
            if meta_offset != cur:
                out_f.write(b"\x00" * (meta_offset - cur))

            meta_payload = _encode_metadata_top_map(meta)
            block_payload_crc = _crc32(meta_payload)
            block_header = (
                _METADATA_BLOCK_MAGIC
                + struct.pack(
                    "<III",
                    _METADATA_BLOCK_VERSION,
                    _METADATA_ENCODING_VERSION,
                    0,
                )
                + struct.pack("<Q", len(meta_payload))
                + struct.pack("<I", block_payload_crc)
                + struct.pack("<I", 0)
            )
            if len(block_header) != 32:
                raise AssertionError("metadata block header size mismatch")
            out_f.write(block_header)
            out_f.write(meta_payload)
            meta_length = 32 + len(meta_payload)

            # Write header (preamble + slots)
            slot_a = _pack_slot(
                generation=1,
                payload_offset=payload_offset,
                payload_length=payload_length,
                metadata_offset=meta_offset,
                metadata_length=meta_length,
                hot_offset=0,
                hot_length=0,
            )
            slot_b = _pack_slot(
                generation=0,
                payload_offset=payload_offset,
                payload_length=payload_length,
                metadata_offset=meta_offset,
                metadata_length=meta_length,
                hot_offset=0,
                hot_length=0,
            )
            out_f.seek(0)
            out_f.write(_pack_preamble())
            out_f.seek(_SLOT_A_OFFSET)
            out_f.write(slot_a)
            out_f.seek(_SLOT_B_OFFSET)
            out_f.write(slot_b)

    finally:
        if temp_raw.exists():
            try:
                temp_raw.unlink()
            except OSError:
                pass


def load(path: str | Path, *, deps: PersistenceDeps) -> Any:
    """Load a matrix or vector from a file (new single-file container only)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not _is_new_container(path):
        raise ValueError("unsupported file format (expected new .pycauset container)")

    metadata, data_offset = _read_new_container_metadata_and_offset(path)

    matrix_type = metadata.get("matrix_type")
    data_type = metadata.get("data_type")

    if matrix_type == "BLOCK":
        try:
            from .blockmatrix import BlockMatrix
        except Exception as e:  # pragma: no cover
            raise ImportError("BlockMatrix support is unavailable") from e
        return _load_blockmatrix(path, metadata, deps=deps, BlockMatrix=BlockMatrix)

    if "rows" not in metadata or "cols" not in metadata:
        raise ValueError("invalid file: missing rows/cols in typed metadata")

    rows = int(metadata["rows"])
    cols = int(metadata["cols"])
    seed = int(metadata.get("seed", 0))

    view = metadata.get("view", {})
    if not isinstance(view, dict):
        view = {}

    scalar = view.get("scalar", 1.0)
    if isinstance(scalar, dict) and "real" in scalar and "imag" in scalar:
        scalar = complex(float(scalar["real"]), float(scalar["imag"]))

    is_transposed = bool(view.get("is_transposed", False))
    is_conjugated = bool(view.get("is_conjugated", False))

    payload_uuid = metadata.get("payload_uuid")
    if not isinstance(payload_uuid, str) or not payload_uuid:
        raise ValueError("invalid file: missing payload_uuid in typed metadata")

    view_sig = _compute_view_signature(
        is_transposed=is_transposed,
        is_conjugated=is_conjugated,
        scalar=scalar,
    )

    def _require_square_dims_for_type(type_name: str) -> None:
        if rows != cols:
            raise ValueError(f"{type_name} requires rows == cols (got rows={rows}, cols={cols})")

    obj = None
    if matrix_type == "CAUSAL" and deps.TriangularBitMatrix is not None:
        _require_square_dims_for_type("CAUSAL")
        obj = deps.TriangularBitMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "IDENTITY" and deps.IdentityMatrix is not None:
        obj = deps.IdentityMatrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "DENSE_FLOAT":
        if data_type == "BIT" and deps.DenseBitMatrix is not None:
            obj = deps.DenseBitMatrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT16" and deps.Float16Matrix is not None:
            obj = deps.Float16Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT32" and deps.Float32Matrix is not None:
            obj = deps.Float32Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT64" and deps.FloatMatrix is not None:
            obj = deps.FloatMatrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT16" and deps.ComplexFloat16Matrix is not None:
            obj = deps.ComplexFloat16Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT32" and deps.ComplexFloat32Matrix is not None:
            obj = deps.ComplexFloat32Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT64" and deps.ComplexFloat64Matrix is not None:
            obj = deps.ComplexFloat64Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "INTEGER":
        if data_type == "INT8" and deps.Int8Matrix is not None:
            obj = deps.Int8Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT16" and deps.Int16Matrix is not None:
            obj = deps.Int16Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT32" and deps.IntegerMatrix is not None:
            obj = deps.IntegerMatrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT64" and deps.Int64Matrix is not None:
            obj = deps.Int64Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT8" and deps.UInt8Matrix is not None:
            obj = deps.UInt8Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT16" and deps.UInt16Matrix is not None:
            obj = deps.UInt16Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT32" and deps.UInt32Matrix is not None:
            obj = deps.UInt32Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT64" and deps.UInt64Matrix is not None:
            obj = deps.UInt64Matrix._from_storage(rows, cols, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "TRIANGULAR_FLOAT" and deps.TriangularFloatMatrix is not None:
        _require_square_dims_for_type("TRIANGULAR_FLOAT")
        obj = deps.TriangularFloatMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "TRIANGULAR_INTEGER" and deps.TriangularIntegerMatrix is not None:
        _require_square_dims_for_type("TRIANGULAR_INTEGER")
        obj = deps.TriangularIntegerMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "VECTOR":
        if data_type == "FLOAT64" and deps.FloatVector is not None:
            obj = deps.FloatVector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT32" and deps.Float32Vector is not None:
            obj = deps.Float32Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT16" and deps.Float16Vector is not None:
            obj = deps.Float16Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT16" and deps.ComplexFloat16Vector is not None:
            obj = deps.ComplexFloat16Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT32" and deps.ComplexFloat32Vector is not None:
            obj = deps.ComplexFloat32Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT64" and deps.ComplexFloat64Vector is not None:
            obj = deps.ComplexFloat64Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT8" and deps.Int8Vector is not None:
            obj = deps.Int8Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT32" and deps.IntegerVector is not None:
            obj = deps.IntegerVector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT16" and deps.Int16Vector is not None:
            obj = deps.Int16Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT64" and deps.Int64Vector is not None:
            obj = deps.Int64Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT8" and deps.UInt8Vector is not None:
            obj = deps.UInt8Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT16" and deps.UInt16Vector is not None:
            obj = deps.UInt16Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT32" and deps.UInt32Vector is not None:
            obj = deps.UInt32Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT64" and deps.UInt64Vector is not None:
            obj = deps.UInt64Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "BIT" and deps.BitVector is not None:
            obj = deps.BitVector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "UNIT_VECTOR" and deps.UnitVector is not None:
        obj = deps.UnitVector._from_storage(rows, seed, str(path), data_offset, seed, scalar, is_transposed)

    if obj is None:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

    if is_transposed and hasattr(obj, "set_transposed"):
        try:
            obj.set_transposed(True)
        except Exception:
            pass

    if is_conjugated and hasattr(obj, "set_conjugated"):
        try:
            obj.set_conjugated(True)
        except Exception:
            pass

    # Restore gospel properties.
    persisted_props = metadata.get("properties")
    if isinstance(persisted_props, dict):
        try:
            obj.properties = persisted_props
        except Exception:
            pass

    cached = metadata.get("cached", {})
    if isinstance(cached, dict):
        trace_entry = cached.get("trace")
        if (
            isinstance(trace_entry, dict)
            and trace_entry.get("value") is not None
            and isinstance(trace_entry.get("signature"), dict)
            and trace_entry["signature"].get("payload_uuid") == payload_uuid
            and trace_entry["signature"].get("view_signature") == view_sig
        ):
            try:
                obj.properties["trace"] = float(trace_entry["value"])
            except Exception:
                pass

        det_entry = cached.get("determinant")
        if (
            isinstance(det_entry, dict)
            and det_entry.get("value") is not None
            and isinstance(det_entry.get("signature"), dict)
            and det_entry["signature"].get("payload_uuid") == payload_uuid
            and det_entry["signature"].get("view_signature") == view_sig
        ):
            try:
                obj.properties["determinant"] = float(det_entry["value"])
            except Exception:
                pass

        norm_entry = cached.get("norm")
        if (
            isinstance(norm_entry, dict)
            and norm_entry.get("value") is not None
            and isinstance(norm_entry.get("signature"), dict)
            and norm_entry["signature"].get("payload_uuid") == payload_uuid
            and norm_entry["signature"].get("view_signature") == view_sig
        ):
            try:
                obj.properties["norm"] = float(norm_entry["value"])
            except Exception:
                pass

        sum_entry = cached.get("sum")
        if (
            isinstance(sum_entry, dict)
            and isinstance(sum_entry.get("value"), list)
            and len(sum_entry.get("value")) == 2
            and isinstance(sum_entry.get("signature"), dict)
            and sum_entry["signature"].get("payload_uuid") == payload_uuid
            and sum_entry["signature"].get("view_signature") == view_sig
        ):
            try:
                r, i = sum_entry["value"]
                obj.properties["sum"] = complex(float(r), float(i))
            except Exception:
                pass

        eigen_entry = cached.get("eigenvalues")
        if (
            isinstance(eigen_entry, dict)
            and eigen_entry.get("value") is not None
            and isinstance(eigen_entry.get("signature"), dict)
            and eigen_entry["signature"].get("payload_uuid") == payload_uuid
            and eigen_entry["signature"].get("view_signature") == view_sig
        ):
            try:
                obj.properties["eigenvalues"] = [complex(r, i) for r, i in eigen_entry["value"]]
            except Exception:
                pass

    if metadata.get("object_type") == "CausalSet":
        st_meta = metadata.get("spacetime", {})
        st_type = st_meta.get("type")
        st_args = st_meta.get("args", {})

        st = None
        if st_type == "MinkowskiDiamond":
            st = deps.native.MinkowskiDiamond(st_args.get("dimension", 2))
        elif st_type == "MinkowskiCylinder":
            st = deps.native.MinkowskiCylinder(
                st_args.get("dimension", 2),
                st_args.get("height", 1.0),
                st_args.get("circumference", 1.0),
            )
        elif st_type == "MinkowskiBox":
            st = deps.native.MinkowskiBox(
                st_args.get("dimension", 2),
                st_args.get("time_extent", 1.0),
                st_args.get("space_extent", 1.0),
            )

        n = int(metadata.get("n", rows))
        return deps.CausalSet(n=n, spacetime=st, seed=seed, matrix=obj)

    return obj


def _save_blockmatrix(
    obj: Any,
    path: Path,
    *,
    deps: PersistenceDeps,
    BlockMatrix: Any,
    ThunkBlock: Any,
) -> None:
    try:
        from .submatrix_view import SubmatrixView
    except Exception:  # pragma: no cover
        SubmatrixView = None  # type: ignore[assignment]

    LazyMatrix = getattr(deps.native, "LazyMatrix", None)

    def _numpy_dtype_hint_for(obj: Any) -> Any | None:
        name = type(obj).__name__
        if name == "Float16Matrix":
            return "float16"
        if name == "Float32Matrix":
            return "float32"
        if name == "FloatMatrix" or name.endswith("FloatMatrix"):
            return "float64"
        if name == "ComplexFloat16Matrix":
            # NumPy has no complex16; fall back to complex64.
            return "complex64"
        if name == "ComplexFloat32Matrix":
            return "complex64"
        if name == "ComplexFloat64Matrix":
            return "complex128"
        if name == "Int8Matrix":
            return "int8"
        if name == "Int16Matrix":
            return "int16"
        if name == "IntegerMatrix":
            return "int32"
        if name == "Int64Matrix":
            return "int64"
        if name == "UInt8Matrix":
            return "uint8"
        if name == "UInt16Matrix":
            return "uint16"
        if name == "UInt32Matrix":
            return "uint32"
        if name == "UInt64Matrix":
            return "uint64"
        if name == "DenseBitMatrix":
            return "bool"
        return None

    def _materialize_view_block(view_obj: Any) -> Any:
        # Materialize to a small numpy array (block-local) and convert to a native matrix.
        asarray = getattr(deps.native, "asarray", None)
        if asarray is None:
            raise TypeError("cannot save SubmatrixView blocks (native.asarray unavailable)")
        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise TypeError("cannot save SubmatrixView blocks (NumPy unavailable)") from e

        src = getattr(view_obj, "source", None)
        dtype_hint = _numpy_dtype_hint_for(src) if src is not None else None
        if dtype_hint is None:
            raise TypeError(
                "cannot save SubmatrixView blocks (cannot infer dtype from view source; materialization policy requires a known dtype)"
            )

        rows = int(view_obj.rows())
        cols = int(view_obj.cols())
        arr = np.empty((rows, cols), dtype=dtype_hint)
        get_fn = getattr(view_obj, "get", None)
        if not callable(get_fn):
            raise TypeError("cannot save SubmatrixView blocks (view has no get(i,j))")
        for i in range(rows):
            for j in range(cols):
                arr[i, j] = get_fn(i, j)
        return asarray(arr)

    blocks_dir = Path(str(path) + ".blocks")
    blocks_dir.mkdir(parents=True, exist_ok=True)

    # Stage all managed children in a temp directory first, then commit via rename/replace.
    # This reduces the chance of leaving a partially updated multi-file tree.
    staging_tag = uuid.uuid4().hex
    staging_dir = blocks_dir / f".staging_{staging_tag}"
    if staging_dir.exists():
        try:
            shutil.rmtree(staging_dir)
        except OSError:
            pass
    staging_dir.mkdir(parents=True, exist_ok=True)

    def _safe_rmtree(p: Path) -> None:
        try:
            shutil.rmtree(p)
        except OSError:
            pass

    def _replace_dir(*, src: Path, dst: Path) -> None:
        if not src.exists():
            return

        old = None
        if dst.exists():
            old = dst.with_name(dst.name + f".old_{staging_tag}")
            try:
                if old.exists():
                    _safe_rmtree(old)
                os.replace(str(dst), str(old))
            except OSError:
                # Fall back to removing dst if rename fails.
                _safe_rmtree(dst)
                old = None

        try:
            os.replace(str(src), str(dst))
        except OSError:
            # Last resort: attempt a non-atomic move.
            if dst.exists():
                _safe_rmtree(dst)
            shutil.move(str(src), str(dst))

        if old is not None:
            _safe_rmtree(old)

    children: list[list[dict[str, Any]]] = []

    expected_child_names: set[str] = set()

    for r in range(obj.block_rows):
        row_entries: list[dict[str, Any]] = []
        for c in range(obj.block_cols):
            blk = obj.get_block(r, c)

            # Evaluate thunks blockwise on save.
            if ThunkBlock is not None and isinstance(blk, ThunkBlock):
                blk = blk.materialize()

            # Materialize lazy expressions so payload storage exists.
            if LazyMatrix is not None and isinstance(blk, LazyMatrix):
                eval_fn = getattr(blk, "eval", None)
                if callable(eval_fn):
                    blk = eval_fn()

            # Persisting a view requires writing stable storage. In Phase E we
            # materialize the view block-locally (no global densification).
            if SubmatrixView is not None and isinstance(blk, SubmatrixView):
                blk = _materialize_view_block(blk)

            child_name = f"block_r{r}_c{c}.pycauset"
            expected_child_names.add(child_name)

            staged_child_path = staging_dir / child_name
            final_child_path = blocks_dir / child_name

            # Save recursively.
            # NOTE: BlockMatrix persistence is multi-file (sidecar directory), so using a
            # temp filename + rename would strand the sidecar directory under the temp name.
            if BlockMatrix is not None and isinstance(blk, BlockMatrix):
                save(blk, staged_child_path, deps=deps)
            else:
                # Best-effort atomic replace for single-file children.
                tmp_child_path = staged_child_path.with_suffix(staged_child_path.suffix + ".tmp")
                if tmp_child_path.exists():
                    try:
                        tmp_child_path.unlink()
                    except OSError:
                        pass
                save(blk, tmp_child_path, deps=deps)
                os.replace(str(tmp_child_path), str(staged_child_path))

            # Pin child payload UUID for snapshot/integrity validation on load.
            try:
                child_meta, _ = _read_new_container_metadata_and_offset(staged_child_path)
                child_uuid = child_meta.get("payload_uuid")
                if isinstance(child_uuid, str) and child_uuid:
                    row_entries.append({"path": child_name, "payload_uuid": child_uuid})
                else:
                    row_entries.append({"path": child_name})
            except Exception:
                row_entries.append({"path": child_name})
        children.append(row_entries)

    # Commit staged children into the real blocks directory.
    for child_name in sorted(expected_child_names):
        staged_child_path = staging_dir / child_name
        final_child_path = blocks_dir / child_name

        # If the staged child is a BlockMatrix container, it has its own sidecar directory.
        staged_child_blocks = Path(str(staged_child_path) + ".blocks")
        final_child_blocks = Path(str(final_child_path) + ".blocks")
        if staged_child_blocks.exists():
            # Commit the sidecar first so the new file immediately points at the correct tree.
            _replace_dir(src=staged_child_blocks, dst=final_child_blocks)

        os.replace(str(staged_child_path), str(final_child_path))

    # Cleanup staged directory.
    _safe_rmtree(staging_dir)

    # Best-effort cleanup of stale managed child files after commit.
    # Preserve unrelated files in the blocks directory.
    try:
        for child in blocks_dir.glob("block_r*_c*.pycauset"):
            if child.name not in expected_child_names:
                try:
                    child.unlink()
                except OSError:
                    pass
                # Also remove managed nested sidecar directory if present.
                nested_dir = Path(str(child) + ".blocks")
                if nested_dir.exists():
                    _safe_rmtree(nested_dir)
    except OSError:
        pass

    # BlockMatrix containers use an empty payload and store structure in metadata.
    meta: dict[str, Any] = {
        "rows": int(obj.rows()),
        "cols": int(obj.cols()),
        "seed": 0,
        "payload_uuid": uuid.uuid4().hex,
        "view": {"is_transposed": False, "is_conjugated": False, "scalar": 1.0},
        "payload_layout": {"kind": "none", "params": {}},
        "matrix_type": "BLOCK",
        "data_type": "MIXED",
        "block_manifest": {
            "version": 1,
            "row_partitions": list(obj.row_partitions),
            "col_partitions": list(obj.col_partitions),
            "children": children,
        },
    }

    # Write file (single-file container) atomically via temp + replace.
    payload_offset = _HEADER_BYTES
    payload_length = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass

    with tmp_path.open("wb") as out_f:
        out_f.write(b"\x00" * _HEADER_BYTES)

        cur = out_f.tell()
        meta_offset = _align_up(cur, 16)
        if meta_offset != cur:
            out_f.write(b"\x00" * (meta_offset - cur))

        meta_payload = _encode_metadata_top_map(meta)
        block_payload_crc = _crc32(meta_payload)
        block_header = (
            _METADATA_BLOCK_MAGIC
            + struct.pack(
                "<III",
                _METADATA_BLOCK_VERSION,
                _METADATA_ENCODING_VERSION,
                0,
            )
            + struct.pack("<Q", len(meta_payload))
            + struct.pack("<I", block_payload_crc)
            + struct.pack("<I", 0)
        )
        if len(block_header) != 32:
            raise AssertionError("metadata block header size mismatch")
        out_f.write(block_header)
        out_f.write(meta_payload)
        meta_length = 32 + len(meta_payload)

        slot_a = _pack_slot(
            generation=1,
            payload_offset=payload_offset,
            payload_length=payload_length,
            metadata_offset=meta_offset,
            metadata_length=meta_length,
            hot_offset=0,
            hot_length=0,
        )
        slot_b = _pack_slot(
            generation=0,
            payload_offset=payload_offset,
            payload_length=payload_length,
            metadata_offset=meta_offset,
            metadata_length=meta_length,
            hot_offset=0,
            hot_length=0,
        )
        out_f.seek(0)
        out_f.write(_pack_preamble())
        out_f.seek(_SLOT_A_OFFSET)
        out_f.write(slot_a)
        out_f.seek(_SLOT_B_OFFSET)
        out_f.write(slot_b)

    os.replace(str(tmp_path), str(path))


def _load_blockmatrix(path: Path, metadata: dict[str, Any], *, deps: PersistenceDeps, BlockMatrix: Any) -> Any:
    manifest = metadata.get("block_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("invalid BLOCK file: missing block_manifest")

    row_parts = manifest.get("row_partitions")
    col_parts = manifest.get("col_partitions")
    children = manifest.get("children")
    if not (isinstance(row_parts, list) and isinstance(col_parts, list) and isinstance(children, list)):
        raise ValueError("invalid BLOCK file: malformed manifest")

    blocks_dir = Path(str(path) + ".blocks")
    if not blocks_dir.exists():
        raise FileNotFoundError(f"missing blocks directory: {blocks_dir}")

    grid: list[list[Any]] = []
    for row in children:
        if not isinstance(row, list):
            raise ValueError("invalid BLOCK file: children grid must be list-of-lists")
        out_row: list[Any] = []
        for entry in row:
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError("invalid BLOCK file: child entry must be a dict with path")
            rel = entry["path"]
            if not isinstance(rel, str) or not rel:
                raise ValueError("invalid BLOCK file: child path must be a non-empty string")
            # Disallow path traversal / absolute references. Child references are filenames
            # within the sibling blocks directory.
            if rel != Path(rel).name or "/" in rel or "\\" in rel or ":" in rel or rel in {".", ".."}:
                raise ValueError("invalid BLOCK file: child path must be a simple filename")
            child_path = blocks_dir / rel

            expected_uuid = entry.get("payload_uuid")
            if expected_uuid is not None:
                if not isinstance(expected_uuid, str) or not expected_uuid:
                    raise ValueError("invalid BLOCK file: child payload_uuid must be a non-empty string")
                try:
                    child_meta, _ = _read_new_container_metadata_and_offset(child_path)
                    actual_uuid = child_meta.get("payload_uuid")
                except Exception as e:
                    raise ValueError("invalid BLOCK file: failed to read child metadata") from e
                if actual_uuid != expected_uuid:
                    raise ValueError("invalid BLOCK file: child payload_uuid mismatch (snapshot violated)")

            out_row.append(load(child_path, deps=deps))
        grid.append(out_row)

    obj = BlockMatrix(grid)
    if list(obj.row_partitions) != list(row_parts) or list(obj.col_partitions) != list(col_parts):
        raise ValueError("invalid BLOCK file: partitions do not match child block shapes")
    return obj
