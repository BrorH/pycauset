from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pycauset import _storage

try:  # resource is not guaranteed on Windows
    import resource as _resource  # type: ignore
except Exception:  # pragma: no cover
    _resource = None


@dataclass
class OperandSnapshot:
    shape: Tuple[int, int] | None
    estimated_bytes: int | None
    is_file_backed: bool
    backing_file: str | None
    is_temporary: bool
    storage_kind: str
    dtype: str | None
    itemsize: int


@dataclass
class IORecord:
    op: str
    route: str
    reason: str
    trace_tag: str
    operands: List[OperandSnapshot]
    tile_shape: Tuple[int, int] | None
    queue_depth: int | None
    throughput: Dict[str, float | None]
    page_faults: int | None
    device_idle: float | None
    events: List[Dict[str, Any]]
    storage: Dict[str, Any]
    timestamp: float


def _get_backing_path(obj: Any) -> Path | None:
    for attr in ("get_backing_file", "backing_file"):
        try:
            candidate = getattr(obj, attr, None)
            if callable(candidate):
                candidate = candidate()
            if candidate:
                return Path(candidate)
        except Exception:
            continue
    return None


def _try_io_prefetch(obj: Any) -> None:
    """Best-effort IO prefetch for objects exposing get_accelerator/backing file."""

    try:
        get_acc = getattr(obj, "get_accelerator", None)
        if not callable(get_acc):
            return
        acc = get_acc()
        if acc is None:
            return

        path = _get_backing_path(obj)
        if path is None:
            return
        try:
            size = int(os.path.getsize(path))
        except OSError:
            return
        if size <= 0:
            return
        acc.prefetch(0, size)
    except Exception:
        return


def _try_io_discard(obj: Any) -> None:
    """Best-effort IO discard for objects exposing get_accelerator/backing file."""

    try:
        get_acc = getattr(obj, "get_accelerator", None)
        if not callable(get_acc):
            return
        acc = get_acc()
        if acc is None:
            return

        path = _get_backing_path(obj)
        if path is None:
            return
        try:
            size = int(os.path.getsize(path))
        except OSError:
            return
        if size <= 0:
            return
        acc.discard(0, size)
    except Exception:
        return


def _discard_if_streaming(record: dict[str, Any] | None, items: list[Any]) -> None:
    """Discard backing ranges when a streaming route was chosen (best-effort)."""

    try:
        if record is None or record.get("route") != "streaming":
            return
    except Exception:
        return

    for obj in items:
        _try_io_discard(obj)


def _dtype_label(obj: Any) -> str | None:
    dtype_attr = getattr(obj, "dtype", None)
    try:
        if dtype_attr is not None and not callable(dtype_attr):
            return str(dtype_attr)
    except Exception:
        pass
    try:
        name = type(obj).__name__
    except Exception:
        return None
    return name


def _shape(obj: Any) -> Tuple[int, int] | None:
    try:
        return int(obj.rows()), int(obj.cols())
    except Exception:
        pass
    try:
        shape_attr = getattr(obj, "shape", None)
        if isinstance(shape_attr, tuple) and len(shape_attr) == 2:
            return int(shape_attr[0]), int(shape_attr[1])
    except Exception:
        pass
    return None


def _itemsize(obj: Any) -> int:
    dtype_attr = getattr(obj, "dtype", None)
    try:
        itemsize_attr = getattr(dtype_attr, "itemsize", None)
        if itemsize_attr is not None:
            return int(itemsize_attr)
    except Exception:
        pass
    try:
        size_attr = getattr(obj, "itemsize", None)
        if size_attr is not None:
            return int(size_attr)
    except Exception:
        pass
    return 8


def _estimate_bytes(obj: Any) -> int | None:
    shp = _shape(obj)
    if shp is None:
        return None
    rows, cols = shp
    try:
        return int(rows * cols * _itemsize(obj))
    except Exception:
        return None


def _page_fault_proxy() -> int | None:
    if _resource is None:
        return None
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        maj = int(getattr(usage, "ru_majflt", 0) or 0)
        minf = int(getattr(usage, "ru_minflt", 0) or 0)
        return maj + minf
    except Exception:
        return None


class IOObservability:
    def __init__(self, *, memory_threshold_bytes: int | None = None) -> None:
        self._memory_threshold = memory_threshold_bytes
        self._counter = 0
        self._last: dict[str, dict[str, Any]] = {}

    def set_memory_threshold(self, value: int | None) -> int | None:
        self._memory_threshold = value
        return self._memory_threshold

    def get_memory_threshold(self) -> int | None:
        return self._memory_threshold

    def clear(self) -> None:
        self._last.clear()

    def _derive_tile_shape(self, operands: List[OperandSnapshot]) -> Tuple[int, int]:
        budget = self._memory_threshold if self._memory_threshold is not None else 1 << 20
        max_itemsize = max((8,) + tuple(op.itemsize for op in operands if op is not None))
        try:
            elems = max(1, int(budget // max(1, max_itemsize)))
            dim = max(1, int(math.sqrt(elems)))
            return dim, dim
        except Exception:
            return 64, 64

    def _record(self, record: IORecord) -> dict[str, Any]:
        payload = asdict(record)
        self._last["__latest__"] = payload
        self._last[record.op] = payload
        return payload

    def _choose_route(self, *, operands: List[OperandSnapshot], allow_huge: bool) -> tuple[str, str]:
        if any(op.is_file_backed for op in operands):
            return "streaming", "file-backed operand"

        if allow_huge:
            return "direct", "allow_huge bypassed threshold"

        threshold = self._memory_threshold
        if threshold is None:
            return "direct", "no threshold configured"

        est = [op.estimated_bytes for op in operands if op.estimated_bytes is not None]
        if any(val > threshold for val in est):
            return "streaming", "estimated bytes exceed threshold"

        return "direct", "below threshold"

    def _snapshot(self, obj: Any) -> OperandSnapshot:
        backing = _get_backing_path(obj)
        shape = _shape(obj)
        estimated = _estimate_bytes(obj)
        dtype = _dtype_label(obj)
        is_file_backed = False
        is_temp = False
        if backing is not None:
            try:
                is_file_backed = backing.exists()
                is_temp = _storage.is_temporary_file(backing)
            except Exception:
                is_file_backed = True
                is_temp = False
        storage_kind = "memory"
        if backing is not None:
            storage_kind = "temp" if is_temp else "file"
        return OperandSnapshot(
            shape=shape,
            estimated_bytes=estimated,
            is_file_backed=is_file_backed,
            backing_file=str(backing) if backing is not None else None,
            is_temporary=is_temp,
            storage_kind=storage_kind,
            dtype=dtype,
            itemsize=_itemsize(obj),
        )

    def _storage_summary(self, snapshots: List[OperandSnapshot]) -> Dict[str, Any]:
        backing_files = [op.backing_file for op in snapshots if op.backing_file]
        temporary = [op.backing_file for op in snapshots if op.is_temporary and op.backing_file]
        roots = []
        for path in backing_files:
            try:
                roots.append(str(Path(path).parent))
            except Exception:
                continue
        return {
            "backing_files": backing_files,
            "temporary_files": temporary,
            "storage_roots": sorted(set(roots)),
            "spilled": any(op.is_temporary for op in snapshots),
        }

    def plan_and_record(
        self,
        op: str,
        operands: List[Any],
        *,
        allow_huge: bool = False,
    ) -> dict[str, Any]:
        snapshots = [self._snapshot(obj) for obj in operands]
        route, reason = self._choose_route(operands=snapshots, allow_huge=allow_huge)

        tile_shape = self._derive_tile_shape(snapshots) if route == "streaming" else None
        queue_depth = 2 if route == "streaming" else 0
        throughput = {"read_mb_s": None, "write_mb_s": None}

        self._counter += 1
        trace_tag = f"{op}:{self._counter}"

        events: List[Dict[str, Any]] = [
            {"type": "plan", "detail": route, "reason": reason},
        ]

        record = IORecord(
            op=op,
            route=route,
            reason=reason,
            trace_tag=trace_tag,
            operands=snapshots,
            tile_shape=tile_shape,
            queue_depth=queue_depth,
            throughput=throughput,
            page_faults=_page_fault_proxy(),
            device_idle=None,
            events=events,
            storage=self._storage_summary(snapshots),
            timestamp=time.time(),
        )
        return self._record(record)

    def last(self, op: str | None = None) -> dict[str, Any] | None:
        key = op or "__latest__"
        payload = self._last.get(key)
        if payload is None:
            return None
        return dict(payload)


# Module-level singleton helpers (optional convenience)
_default_observability = IOObservability()


def default_instance() -> IOObservability:
    return _default_observability
