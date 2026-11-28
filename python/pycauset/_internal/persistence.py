from __future__ import annotations

import os
import shutil
import struct
import tempfile
import uuid
import zlib
from pathlib import Path
from typing import Any, Protocol


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

# On-disk payload layout per structure (metadata "payload_layout.kind").
_PAYLOAD_LAYOUT = {
    "dense": "raw_dense",
    "vector": "raw_dense",
    "unit_vector": "raw_dense",
    "identity": "raw_dense",
    "triangular": "raw_triangular",
    "symmetric": "raw_triangular",
    "antisymmetric": "raw_triangular",
    "hermitian": "raw_triangular",
    "skew_hermitian": "raw_triangular",
    "diagonal": "raw_diagonal",
}



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


def _read_new_container_metadata_and_offset(path: Path) -> tuple[dict[str, Any], int]:
    """The active slot's metadata and payload offset (one header reader, used by load)."""
    active, _, typed_meta = _read_active_slot_and_typed_metadata(path)
    if not isinstance(typed_meta, dict):
        raise ValueError("invalid file: metadata top-level is not a map")
    return typed_meta, int(active["payload_offset"])


class PersistenceDeps(Protocol):
    CausalSet: type
    native: Any  # the native module


def _engine_stamp(native: Any) -> str:
    """Identifies the engine build that computed saved derived values (review 3.9 C-3). Values
    from another build are not trusted: a fixed kernel must not keep serving old answers. The
    package version alone does not change in a source checkout, so the native module's size and
    modification time are part of it (a rebuild or reinstall drops saved values: they are
    recomputed on the next call)."""
    try:
        from .._version import version
    except ImportError:
        version = "unknown"
    native_file = getattr(native, "__file__", None)
    if not native_file:
        return f"{version}|unknown"
    st = os.stat(native_file)
    return f"{version}|{st.st_size}|{st.st_mtime_ns}"


def _staging_file(directory: Path, prefix: str, suffix: str) -> Path:
    """A new, uniquely named file in `directory` (two saves never share one, review 3.9 C-1)."""
    fd, name = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(name)


def _fsync_dir(directory: Path) -> None:
    """Make a rename in `directory` durable (POSIX; Windows has no directory handles)."""
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(obj: Any, path: str | Path, *, deps: PersistenceDeps) -> None:
    """Save a matrix or vector to a file (new single-file format)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Internal BlockMatrix persistence (Phase E).
    # This is not part of the public API surface; it exists to support the
    # block-matrix roadmap without introducing a new file format.
    from .blockmatrix import BlockMatrix  # local import to avoid cycles
    from .thunks import ThunkBlock

    if isinstance(obj, BlockMatrix):
        _save_blockmatrix(obj, path, deps=deps, BlockMatrix=BlockMatrix, ThunkBlock=ThunkBlock)
        return

    causet_obj: Any | None = None
    payload_obj = obj
    if isinstance(obj, deps.CausalSet):  # its payload is the causal matrix
        causet_obj = obj
        payload_obj = obj.causal_matrix

    # Closure views (LAZY.md L2b) read another object's storage; save their packed result.
    if isinstance(payload_obj, deps.native.MatrixBase):
        payload_obj = deps.native._materialize_for_save(payload_obj)
    if not hasattr(payload_obj, "copy_storage"):
        raise TypeError("Object does not support saving (missing copy_storage)")

    # Staging files are unique to this call and are the only files it ever deletes (the fixed
    # names `<stem>.raw_tmp` / `<name>.part` let two saves swap payloads, and saving to
    # "x.raw_tmp" deleted the result, review 3.9 C-1).
    temp_raw = _staging_file(path.parent, f".{path.name}.", ".raw_tmp")
    part: Path | None = None
    try:
        payload_obj.copy_storage(str(temp_raw))
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

        # Derived values (determinant, sums, norms, ...; review DERIVED_CACHE.md D.3) of the
        # storage being saved, keyed by view: the payload is that storage's bytes as they are.
        # (Windows and closures were written out above into a new object, which has none.)
        if isinstance(payload_obj, deps.native.MatrixBase):
            derived = [list(e) for e in deps.native._derived_cache_export(payload_obj)]
            if derived:
                meta["derived"] = derived
                meta["derived_engine"] = _engine_stamp(deps.native)

        if causet_obj is not None:
            meta["object_type"] = "CausalSet"
            meta["n"] = int(causet_obj.n)
            meta["spacetime"] = {"type": "MinkowskiDiamond", "args": {"dimension": causet_obj.spacetime.dimension()}}
            meta["seed"] = int(causet_obj._seed)

        # Structure and dtype come straight from the object (one class per structure).
        structure = deps.native._structure_of(payload_obj)
        meta["structure"] = structure
        meta["dtype"] = str(payload_obj.dtype)
        meta["payload_layout"]["kind"] = _PAYLOAD_LAYOUT[structure]
        if structure == "triangular":
            # The packed layout depends on it: a diagonal-including triangle loaded as strict
            # read shifted values (F-076).
            meta["has_diagonal"] = bool(payload_obj.has_diagonal)


        # Write file: header placeholder, then payload, then metadata block, then header slots.
        payload_offset = _HEADER_BYTES
        raw_header = _raw_storage_header_bytes(temp_raw)
        raw_size = temp_raw.stat().st_size
        payload_length = raw_size - raw_header
        if payload_length < 0:
            raise ValueError("invalid raw payload size")

        # Write a sibling file, then rename it over `path` (atomic): an object loaded from
        # `path` keeps reading its (copy-on-write) mapping of the old file, and a crash never
        # leaves a half-written container (F-093).
        part = _staging_file(path.parent, f".{path.name}.", ".part")
        with part.open("wb") as out_f:
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
            # The container's bytes reach the disk before it replaces the old file, and the
            # rename itself after (only the staging copy was synced, review 3.9 C-2).
            out_f.flush()
            os.fsync(out_f.fileno())
        os.replace(part, path)
        part = None
        _fsync_dir(path.parent)

    finally:
        for staged in (part, temp_raw):
            if staged is not None:
                try:
                    staged.unlink()
                except FileNotFoundError:
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

    if matrix_type == "BLOCK":
        from .blockmatrix import BlockMatrix
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


    # The header records the *logical* shape plus the view flags, while the rectangular
    # `_from_storage` constructors take the *storage* shape and then apply the transpose.
    # Swap for transposed views, otherwise A.T loaded back as A (F-037).
    storage_rows, storage_cols = (cols, rows) if is_transposed else (rows, cols)

    structure = metadata.get("structure")
    dtype = metadata.get("dtype")
    if structure not in _PAYLOAD_LAYOUT or not isinstance(dtype, str):
        raise ValueError(f"invalid file: unknown structure/dtype {structure!r}/{dtype!r}")
    if structure in ("vector", "unit_vector"):
        storage_rows, storage_cols = rows, 1
    elif structure not in ("dense", "identity") and rows != cols:
        raise ValueError(f"{structure} matrix requires rows == cols (got rows={rows}, cols={cols})")
    obj = deps.native._open_storage(
        structure, dtype, storage_rows, storage_cols, str(path), data_offset, seed, scalar, is_transposed,
        bool(metadata.get("has_diagonal", False)),
    )

    if is_transposed and hasattr(obj, "set_transposed"):
        obj.set_transposed(True)

    if is_conjugated and hasattr(obj, "set_conjugated"):
        obj.set_conjugated(True)

    derived = metadata.get("derived")
    if derived and isinstance(obj, deps.native.MatrixBase) and metadata.get("derived_engine") == _engine_stamp(deps.native):
        deps.native._derived_cache_import(obj, derived)


    if metadata.get("object_type") == "CausalSet":
        from ..spacetime import MinkowskiDiamond

        st = metadata.get("spacetime", {})
        if st.get("type") != "MinkowskiDiamond":
            raise ValueError(f"unsupported spacetime in file: {st.get('type')!r}")
        # Written by save(), which only saves valid orders: no O(n^3) validation on load.
        return deps.CausalSet(n=int(metadata["n"]), spacetime=MinkowskiDiamond(int(st["args"]["dimension"])),
                              seed=int(metadata["seed"]), matrix=obj, validate=False)
    return obj


def _save_blockmatrix(
    obj: Any,
    path: Path,
    *,
    deps: PersistenceDeps,
    BlockMatrix: Any,
    ThunkBlock: Any,
) -> None:
    from .submatrix_view import SubmatrixView


    def _numpy_dtype_hint_for(obj: Any) -> Any | None:
        dt = getattr(obj, "dtype", None)
        # NumPy has no complex16; complex_float16 blocks materialize as complex64.
        return {"complex_float16": "complex64", "complex_float32": "complex64",
                "complex_float64": "complex128"}.get(dt, dt)

    def _materialize_view_block(view_obj: Any) -> Any:
        # Materialize to a small numpy array (block-local) and convert to a native matrix.
        asarray = deps.native.asarray
        import numpy as np

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

        # Renames only; a failure raises (it fell back to deleting the old tree and a
        # non-atomic move, review 3.9 C-10). Only removing the replaced tree is best effort.
        old = None
        if dst.exists():
            old = dst.with_name(dst.name + f".old_{staging_tag}")
            os.replace(str(dst), str(old))
        os.replace(str(src), str(dst))
        if old is not None:
            _safe_rmtree(old)

    children: list[list[dict[str, Any]]] = []

    expected_child_names: set[str] = set()

    for r in range(obj.block_rows):
        row_entries: list[dict[str, Any]] = []
        for c in range(obj.block_cols):
            blk = obj.get_block(r, c)

            # Evaluate thunks blockwise on save.
            if isinstance(blk, ThunkBlock):
                blk = blk.materialize()


            # Persisting a view requires writing stable storage. In Phase E we
            # materialize the view block-locally (no global densification).
            if isinstance(blk, SubmatrixView):
                blk = _materialize_view_block(blk)

            child_name = f"block_r{r}_c{c}.pycauset"
            expected_child_names.add(child_name)

            staged_child_path = staging_dir / child_name
            final_child_path = blocks_dir / child_name

            # Save recursively.
            # NOTE: BlockMatrix persistence is multi-file (sidecar directory), so using a
            # temp filename + rename would strand the sidecar directory under the temp name.
            if isinstance(blk, BlockMatrix):
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

            # Pin child payload UUID for snapshot/integrity validation on load. The child
            # was just written by `save`: failing to read it back is an error, not a reason to
            # store the block without its integrity pin.
            child_meta, _ = _read_new_container_metadata_and_offset(staged_child_path)
            child_uuid = child_meta.get("payload_uuid")
            if isinstance(child_uuid, str) and child_uuid:
                row_entries.append({"path": child_name, "payload_uuid": child_uuid})
            else:
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
    tmp_path = _staging_file(path.parent, f".{path.name}.", ".part")  # unique (C-1)

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
        out_f.flush()
        os.fsync(out_f.fileno())

    os.replace(str(tmp_path), str(path))
    _fsync_dir(path.parent)


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
