from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from . import io_observability

TileFn = Callable[[int | None, List[io_observability.OperandSnapshot]], Tuple[int, int]]
QueueFn = Callable[[str, List[io_observability.OperandSnapshot]], int]
GuardFn = Callable[[List[Any], List[io_observability.OperandSnapshot], bool], Tuple[str, str] | None]
PrefetchFn = Callable[[List[Any], dict[str, Any]], None]
DiscardFn = Callable[[List[Any], dict[str, Any], Any | None], None]


@dataclass
class StreamingDescriptor:
    op: str
    access_pattern: str | None = None
    tile_budget_fn: TileFn | None = None
    queue_depth_fn: QueueFn | None = None
    guard: GuardFn | None = None
    prefetch: PrefetchFn | None = None
    discard: DiscardFn | None = None


def _append_event(record: dict[str, Any], *, event_type: str, detail: str, reason: str | None = None) -> None:
    events = record.setdefault("events", [])
    payload: dict[str, Any] = {"type": event_type, "detail": detail}
    if reason is not None:
        payload["reason"] = reason
    events.append(payload)


def _default_tile_shape(threshold_bytes: int | None, operands: List[io_observability.OperandSnapshot]) -> Tuple[int, int]:
    budget = threshold_bytes if threshold_bytes is not None else 1 << 20
    
    sizes = [8]
    for op in operands:
        if op is None: continue
        if isinstance(op, dict):
            sizes.append(op.get("itemsize", 8))
        else:
            try:
                sizes.append(op.itemsize)
            except AttributeError:
                sizes.append(8)
    
    max_itemsize = max(sizes)
    try:
        elems = max(1, int(budget // max(1, max_itemsize)))
        dim = max(1, int(elems**0.5))
        return dim, dim
    except Exception:
        return 64, 64


def _coerce_queue_depth(queue_depth: int | None, route: str) -> int:
    if route != "streaming":
        return 0
    try:
        depth = int(queue_depth)
    except Exception:
        depth = 2
    if depth < 1:
        depth = 1
    if depth > 8:
        depth = 8
    return depth


def _try_prefetch_default(operands: List[Any]) -> None:
    for obj in operands:
        io_observability._try_io_prefetch(obj)


def _try_discard_default(items: List[Any]) -> None:
    for obj in items:
        io_observability._try_io_discard(obj)


def _shape_from_snapshot(snapshot: io_observability.OperandSnapshot | None) -> tuple[int, int] | None:
    try:
        if snapshot is None:
            return None
        if isinstance(snapshot, dict):
            return snapshot.get("shape")
        return snapshot.shape
    except Exception:
        return None


def _max_itemsize(operands: List[io_observability.OperandSnapshot]) -> int:
    sizes = []
    for op in operands:
        if op is None:
            continue
        if isinstance(op, dict):
            sizes.append(op.get("itemsize", 8))
        else:
            try:
                sizes.append(op.itemsize)
            except AttributeError:
                sizes.append(8)
                
    if not sizes:
        return 8
    try:
        return int(max(sizes))
    except Exception:
        return 8


def _matmul_guard(
    operands: List[Any], snapshots: List[io_observability.OperandSnapshot], allow_huge: bool
) -> Tuple[str, str] | None:
    if len(snapshots) >= 2:
        a_shape = _shape_from_snapshot(snapshots[0])
        b_shape = _shape_from_snapshot(snapshots[1])
        if a_shape is not None and b_shape is not None and a_shape[1] != b_shape[0]:
            return "direct", "shape_mismatch"
    return None


def _square_guard(
    operands: List[Any], snapshots: List[io_observability.OperandSnapshot], allow_huge: bool
) -> Tuple[str, str] | None:
    snap = snapshots[0] if snapshots else None
    shape = _shape_from_snapshot(snap)
    if shape is None:
        return None
    if shape[0] != shape[1]:
        return "direct", "non_square"
    return None


def _matmul_tile_budget(
    threshold_bytes: int | None, operands: List[io_observability.OperandSnapshot]
) -> Tuple[int, int] | None:
    if threshold_bytes is None or len(operands) < 2:
        return None

    itemsize = _max_itemsize(operands)
    try:
        budget_elems = max(1, int(threshold_bytes // max(1, itemsize * 2)))
        dim = max(1, int(budget_elems**0.5))
    except Exception:
        dim = 64

    a_rows = _shape_from_snapshot(operands[0])
    b_cols_shape = _shape_from_snapshot(operands[1])
    a_dim = a_rows[0] if a_rows is not None else dim
    b_dim = b_cols_shape[1] if b_cols_shape is not None else dim
    return max(1, min(dim, a_dim)), max(1, min(dim, b_dim))


def _square_tile_budget(
    threshold_bytes: int | None, operands: List[io_observability.OperandSnapshot]
) -> Tuple[int, int] | None:
    base = _default_tile_shape(threshold_bytes, operands)
    snap = operands[0] if operands else None
    shape = _shape_from_snapshot(snap)
    if shape is None or base is None:
        return base
    dim = min(base[0], shape[0])
    return dim, dim


def _matmul_queue_depth(route: str, operands: List[io_observability.OperandSnapshot]) -> int:
    if route != "streaming":
        return 0
    return 3 if len(operands) > 1 else 2


def _unary_queue_depth(route: str, operands: List[io_observability.OperandSnapshot]) -> int:
    if route != "streaming":
        return 0
    return 1


class StreamingManager:
    def __init__(self, *, io_observer: io_observability.IOObservability | None = None) -> None:
        self._io = io_observer or io_observability.default_instance()
        self._registry: Dict[str, StreamingDescriptor] = {}

    def register(self, descriptor: StreamingDescriptor) -> None:
        self._registry[descriptor.op] = descriptor

    def get(self, op: str) -> StreamingDescriptor | None:
        return self._registry.get(op)

    def plan(
        self,
        op: str,
        operands: List[Any],
        *,
        allow_huge: bool = False,
        hints: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        record = self._io.plan_and_record(op, operands, allow_huge=allow_huge)
        if not isinstance(record, dict):
            return record

        descriptor = self._registry.get(op)
        snapshots = record.get("operands", [])

        if descriptor is not None:
            guard = descriptor.guard
            if guard is not None:
                try:
                    decision = guard(operands, snapshots, allow_huge)
                except Exception:
                    decision = None
                if decision is not None:
                    route, reason = decision
                    record["route"] = route
                    record["reason"] = reason
                    _append_event(record, event_type="plan", detail=route, reason=reason)

            if descriptor.access_pattern:
                plan_section = record.setdefault("plan", {})
                plan_section["access_pattern"] = descriptor.access_pattern

        route = record.get("route")
        if route == "streaming":
            threshold = self._io.get_memory_threshold()
            tile_shape = record.get("tile_shape")
            if descriptor is not None and descriptor.tile_budget_fn is not None:
                try:
                    tile_shape = descriptor.tile_budget_fn(threshold, snapshots)
                except Exception:
                    tile_shape = record.get("tile_shape")
            record["tile_shape"] = tile_shape or _default_tile_shape(threshold, snapshots)

            queue_depth = record.get("queue_depth")
            if descriptor is not None and descriptor.queue_depth_fn is not None:
                try:
                    queue_depth = descriptor.queue_depth_fn(route, snapshots)
                except Exception:
                    queue_depth = record.get("queue_depth")
            record["queue_depth"] = _coerce_queue_depth(queue_depth, route)
        else:
            record["tile_shape"] = None
            record["queue_depth"] = 0

        return record

    def prefetch(self, record: dict[str, Any] | None, operands: List[Any]) -> None:
        if not self._is_streaming(record):
            return
        descriptor = self._registry.get(record.get("op")) if isinstance(record, dict) else None
        if descriptor is not None and descriptor.prefetch is not None:
            try:
                descriptor.prefetch(operands, record)  # type: ignore[arg-type]
            except Exception:
                _try_prefetch_default(operands)
        else:
            _try_prefetch_default(operands)
        _append_event(record, event_type="io", detail="prefetch")

    def discard(self, record: dict[str, Any] | None, operands: List[Any], result: Any | None = None) -> None:
        if not self._is_streaming(record):
            return
        payloads = list(operands)
        if result is not None:
            payloads.append(result)
        descriptor = self._registry.get(record.get("op")) if isinstance(record, dict) else None
        if descriptor is not None and descriptor.discard is not None:
            try:
                descriptor.discard(payloads, record, result)  # type: ignore[arg-type]
            except Exception:
                _try_discard_default(payloads)
        else:
            _try_discard_default(payloads)
        _append_event(record, event_type="io", detail="discard")

    def annotate_impl(self, record: dict[str, Any] | None, label: str) -> None:
        if not isinstance(record, dict):
            return
        record["impl"] = label
        _append_event(record, event_type="compute", detail=f"impl={label}")

    @staticmethod
    def _is_streaming(record: dict[str, Any] | None) -> bool:
        return isinstance(record, dict) and record.get("route") == "streaming"


def default_manager() -> StreamingManager:
    return StreamingManager(io_observer=io_observability.default_instance())
