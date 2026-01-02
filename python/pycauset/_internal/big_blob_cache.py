from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from . import persistence as _persistence
from .warnings import PyCausetStorageWarning


_WARNED_CACHE_KEYS: set[tuple[str, str]] = set()


def compute_view_signature(obj: Any) -> str:
    scalar = getattr(obj, "scalar", 1.0)
    if isinstance(scalar, complex):
        sr = float(scalar.real)
        si = float(scalar.imag)
    else:
        sr = float(scalar)
        si = 0.0

    is_transposed = getattr(obj, "is_transposed", False)
    if callable(is_transposed):
        is_transposed = is_transposed()
    is_conjugated = getattr(obj, "is_conjugated", False)
    if callable(is_conjugated):
        is_conjugated = is_conjugated()

    return f"t={int(bool(is_transposed))};c={int(bool(is_conjugated))};sr={sr:g};si={si:g}"


def _warn_storage_once(key: tuple[str, str], message: str) -> None:
    if key in _WARNED_CACHE_KEYS:
        return
    _WARNED_CACHE_KEYS.add(key)
    warnings.warn(message, PyCausetStorageWarning, stacklevel=3)


def try_resolve_cached_object_path(
    base_path: str | Path,
    *,
    name: str,
    view_signature: str | None = None,
) -> Path | None:
    ref = _persistence.try_get_cached_big_blob_ref(base_path, name=name, view_signature=view_signature)
    if not isinstance(ref, dict):
        return None

    ref_kind = ref.get("ref_kind")
    object_id = ref.get("object_id")
    if ref_kind != "sibling_object_store" or not isinstance(object_id, str) or not object_id:
        _warn_storage_once(
            (str(base_path), f"cached.{name}.unsupported_ref"),
            f"cached {name} reference kind unsupported; cache entry ignored (no implicit recompute)",
        )
        return None

    obj_path = _persistence.object_store_path_for_id(base_path, object_id=object_id)
    if not obj_path.exists():
        _warn_storage_once(
            (str(base_path), f"cached.{name}.miss:{object_id}"),
            f"cached {name} object missing; cache entry ignored (object_id={object_id})",
        )
        return None

    return obj_path


def try_load_cached_matrix(
    base_path: str | Path,
    *,
    name: str,
    view_signature: str,
    MatrixClass: Any,
) -> Any | None:
    obj_path = try_resolve_cached_object_path(base_path, name=name, view_signature=view_signature)
    if obj_path is None:
        return None

    try:
        meta, payload_offset = _persistence._read_new_container_metadata_and_offset(obj_path)
        rows = int(meta.get("rows", 0))
        cols = int(meta.get("cols", 0))
        seed = int(meta.get("seed", 0))

        view = meta.get("view", {})
        if not isinstance(view, dict):
            view = {}

        scalar = view.get("scalar", 1.0)
        if isinstance(scalar, dict) and "real" in scalar and "imag" in scalar:
            scalar = complex(float(scalar["real"]), float(scalar["imag"]))

        is_transposed = bool(view.get("is_transposed", False))

        return MatrixClass._from_storage(rows, cols, str(obj_path), int(payload_offset), seed, scalar, is_transposed)
    except Exception:
        # Treat as cache miss: warn and let caller decide what to do.
        # Policy: do not implicitly recompute missing/unreadable cached objects.
        # Best-effort to include object_id in the key/message.
        try:
            ref = _persistence.try_get_cached_big_blob_ref(base_path, name=name, view_signature=view_signature)
            object_id = ref.get("object_id") if isinstance(ref, dict) else None
        except Exception:
            object_id = None

        suffix = f":{object_id}" if object_id else ""
        _warn_storage_once(
            (str(base_path), f"cached.{name}.error{suffix}"),
            f"cached {name} object unreadable; cache entry ignored" + (f" (object_id={object_id})" if object_id else ""),
        )
        return None


def persist_cached_object(
    base_path: str | Path,
    *,
    name: str,
    obj: Any,
    view_signature: str,
) -> str:
    """Persist obj as an independent .pycauset object and link it under cached.<name>.

    Returns the generated object_id.

    Best-effort durability: write object to a temp name and atomically publish via rename
    before committing the reference into the base snapshot.
    """
    object_id = _persistence.new_object_id()
    obj_path = _persistence.object_store_path_for_id(base_path, object_id=object_id)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = obj_path.with_suffix(obj_path.suffix + ".tmp")

    if not hasattr(obj, "save"):
        raise TypeError("cached big-blob object must support save(path)")

    obj.save(str(tmp_path))
    os.replace(tmp_path, obj_path)

    base_meta = _persistence.read_typed_metadata(base_path)
    payload_uuid = base_meta.get("payload_uuid")
    if not isinstance(payload_uuid, str) or not payload_uuid:
        raise ValueError("invalid base snapshot: missing payload_uuid")

    _persistence.write_cached_big_blob_ref(
        base_path,
        name=name,
        ref_kind="sibling_object_store",
        object_id=object_id,
        signature={"payload_uuid": payload_uuid, "view_signature": view_signature},
    )

    return object_id
