from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def temporary_native_memory_threshold(native: Any, limit_bytes: int | None) -> Iterator[None]:
    """Objects created on this thread inside the block are placed by `limit_bytes` (0: in a
    file). Other threads and the global threshold are not affected: the override is the
    engine's per-thread one (it was the global threshold, review 3.8 C-5)."""
    if limit_bytes is None:
        yield
        return
    previous = native._set_thread_memory_threshold(int(limit_bytes))
    try:
        yield
    finally:
        native._set_thread_memory_threshold(int(previous))
