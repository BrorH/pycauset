"""IO traces: what an operation read and how the engine ran it (review F-100).

One trace is written around each traced ``pycauset.<op>(...)`` call. It holds facts only:

- ``operands``: shape, dtype and storage (``"memory"``, ``"file"`` or ``"temp"``) of each input;
- ``kernels``: every kernel the engine ran for the call, in order (the engine's kernel log);
- ``streamed``: whether the engine ran any part of it out of core. That is the engine's own
  decision (``CpuDevice`` -> ``StreamingManager``, from the MemoryGovernor's RAM budget); the
  StreamingManager labels its tile loops ``streaming.*``.

Nothing here plans, routes or tunes anything.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pycauset import _storage


def _facts(obj: Any) -> tuple[Any, Any, Any]:
    """What a trace keeps of an operand at call time: shape, dtype and backing file, raw.
    Formatting them waits for `last()` (each traced call paid ~1 us for it, F-096)."""
    try:
        return obj.shape, obj.dtype, obj.get_backing_file()
    except AttributeError:
        return getattr(obj, "shape", None), getattr(obj, "dtype", None), None


def _snapshot(facts: tuple[Any, Any, Any]) -> dict[str, Any]:
    shape, dtype, backing = facts
    if not backing or backing == ":memory:":
        storage, backing = "memory", None
    else:
        storage = "temp" if _storage.is_temporary_file(Path(backing)) else "file"
    return {
        "shape": tuple(int(x) for x in shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
        "storage": storage,
        "backing_file": backing,
    }


class IOObservability:
    def __init__(self, native: Any) -> None:
        self._native = native
        # op -> (op, operand facts, kernels, timestamp); the record dict is built by last()
        self._last: dict[str, tuple[str, list[Any], list[str], float]] = {}
        self._depth = 0  # a traced op calling another traced op: only the outer one logs

    def clear(self) -> None:
        self._last.clear()

    def last(self, op: str | None = None) -> dict[str, Any] | None:
        raw = self._last.get(op or "__latest__")
        if raw is None:
            return None
        name, facts, kernels, stamp = raw
        return {
            "op": name,
            "operands": [_snapshot(f) for f in facts],
            "kernels": list(kernels),
            "streamed": any(k.startswith("streaming.") for k in kernels),
            "timestamp": stamp,
        }

    def trace(self, op: str, operands: list[Any]) -> "_Trace":
        """Context manager recording `op`'s operands and the kernels the engine ran. A class,
        not a generator: it wraps every traced call (per-call overhead, F-096)."""
        return _Trace(self, op, operands)


class _Trace:
    __slots__ = ("obs", "op", "operands", "facts", "stamp", "outer")

    def __init__(self, obs: IOObservability, op: str, operands: list[Any]) -> None:
        self.obs, self.op, self.operands = obs, op, operands
        self.outer = False

    def __enter__(self) -> None:
        obs = self.obs
        if obs._depth:  # a traced op calling another traced op: only the outer one logs
            return None
        self.outer = True
        self.facts = [_facts(o) for o in self.operands]
        self.stamp = time.time()
        obs._depth += 1
        obs._native._debug_kernel_log_start()
        return None

    def __exit__(self, *exc: Any) -> bool:
        if not self.outer:
            return False
        obs = self.obs
        kernels = obs._native._debug_kernel_log_stop()
        obs._depth -= 1
        raw = (self.op, self.facts, kernels, self.stamp)
        obs._last[self.op] = raw
        obs._last["__latest__"] = raw
        return False
