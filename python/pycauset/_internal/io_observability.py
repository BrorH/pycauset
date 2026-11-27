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


def _snapshot(obj: Any) -> dict[str, Any]:
    shape = getattr(obj, "shape", None)
    backing = obj.get_backing_file() if hasattr(obj, "get_backing_file") else None
    if not backing or backing == ":memory:":
        storage, backing = "memory", None
    else:
        storage = "temp" if _storage.is_temporary_file(Path(backing)) else "file"
    dtype = getattr(obj, "dtype", None)
    return {
        "shape": tuple(int(x) for x in shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
        "storage": storage,
        "backing_file": backing,
    }


class IOObservability:
    def __init__(self, native: Any) -> None:
        self._native = native
        self._last: dict[str, dict[str, Any]] = {}
        self._depth = 0  # a traced op calling another traced op: only the outer one logs

    def clear(self) -> None:
        self._last.clear()

    def last(self, op: str | None = None) -> dict[str, Any] | None:
        rec = self._last.get(op or "__latest__")
        return dict(rec) if rec is not None else None

    def trace(self, op: str, operands: list[Any]) -> "_Trace":
        """Context manager recording `op`'s operands and the kernels the engine ran. A class,
        not a generator: it wraps every traced call (per-call overhead, F-096)."""
        return _Trace(self, op, operands)


class _Trace:
    __slots__ = ("obs", "op", "operands", "rec", "outer")

    def __init__(self, obs: IOObservability, op: str, operands: list[Any]) -> None:
        self.obs, self.op, self.operands = obs, op, operands
        self.rec: dict[str, Any] = {}
        self.outer = False

    def __enter__(self) -> dict[str, Any]:
        obs = self.obs
        if obs._depth:  # a traced op calling another traced op: only the outer one logs
            return self.rec
        self.outer = True
        self.rec = {
            "op": self.op,
            "operands": [_snapshot(o) for o in self.operands],
            "kernels": [],
            "streamed": False,
            "timestamp": time.time(),
        }
        obs._depth += 1
        obs._native._debug_kernel_log_start()
        return self.rec

    def __exit__(self, *exc: Any) -> bool:
        if not self.outer:
            return False
        obs, rec = self.obs, self.rec
        kernels = list(obs._native._debug_kernel_log_stop())
        obs._depth -= 1
        rec["kernels"] = kernels
        rec["streamed"] = any(k.startswith("streaming.") for k in kernels)
        obs._last[self.op] = rec
        obs._last["__latest__"] = rec
        return False
