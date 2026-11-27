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

    prev = native.get_memory_threshold()
    native.set_memory_threshold(int(limit_bytes))
    try:
        yield
    finally:
        native.set_memory_threshold(int(prev))
