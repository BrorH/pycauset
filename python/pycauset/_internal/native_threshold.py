from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def temporary_native_memory_threshold(native: Any, limit_bytes: int | None) -> Iterator[None]:
    """Temporarily override the native memory threshold (best-effort).

    This is used to implement per-call import behavior like max_in_ram_bytes
    without permanently changing global process settings.

    Note: the underlying native threshold is process-global; this helper is
    best-effort and not intended to provide strict thread isolation.
    """

    if limit_bytes is None:
        yield
        return

    setter = getattr(native, "set_memory_threshold", None)
    getter = getattr(native, "get_memory_threshold", None)
    if not callable(setter) or not callable(getter):
        yield
        return

    try:
        prev = getter()
    except Exception:
        prev = None

    try:
        setter(int(limit_bytes))
        yield
    finally:
        try:
            if prev is not None:
                setter(int(prev))
        except Exception:
            pass
