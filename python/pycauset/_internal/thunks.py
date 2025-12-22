from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import threading


class StaleThunkError(RuntimeError):
    pass


def _get_version(obj: Any) -> int | None:
    v = getattr(obj, "version", None)
    try:
        if v is None:
            return None
        if callable(v):
            return int(v())
        return int(v)
    except Exception:
        return None


@dataclass(frozen=True)
class _VersionPin:
    obj: Any
    version: int


class ThunkBlock:
    """Lazy block that evaluates to a concrete matrix-like object.

    Phase D scope:
    - Known shape at construction time
    - Cached evaluation
    - Staleness check based on pinned `version` attributes on captured sources

    Evaluation triggers:
    - element access (`get` / `__getitem__`)
    - any explicit materialize

    Non-triggers:
    - repr/str
    """

    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        compute: Callable[[], Any],
        sources_for_staleness: Iterable[Any] = (),
        label: str = "thunk",
    ):
        self._rows = int(rows)
        self._cols = int(cols)
        self._compute = compute
        self._label = str(label)

        pins: list[_VersionPin] = []
        for src in sources_for_staleness:
            v = _get_version(src)
            if v is not None:
                pins.append(_VersionPin(obj=src, version=v))
        self._pins = tuple(pins)

        self._cache: Any | None = None
        self._lock = threading.Lock()

    def rows(self) -> int:
        return self._rows

    def cols(self) -> int:
        return self._cols

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows(), self.cols())

    @property
    def dtype(self) -> str:
        # Structure-only label; heterogeneous containers use MIXED anyway.
        return self._label

    def _check_stale(self) -> None:
        for pin in self._pins:
            cur = _get_version(pin.obj)
            if cur is None:
                continue
            if cur != pin.version:
                raise StaleThunkError(
                    f"stale thunk: input {type(pin.obj).__name__} version changed ({pin.version} -> {cur})"
                )

    def materialize(self) -> Any:
        # Staleness is checked on every access (including cache hits).
        self._check_stale()
        if self._cache is not None:
            return self._cache

        # Ensure single-eval under concurrency.
        with self._lock:
            self._check_stale()
            if self._cache is None:
                self._cache = self._compute()
            return self._cache

    def get(self, i: int, j: int) -> Any:
        m = self.materialize()
        get_fn = getattr(m, "get", None)
        if callable(get_fn):
            return get_fn(i, j)
        return m[i, j]

    def __getitem__(self, key: Any) -> Any:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("matrix indices must be provided as [row, col]")
        i, j = key
        return self.get(int(i), int(j))

    def __repr__(self) -> str:
        return f"ThunkBlock(shape={self.shape}, label={self._label})"

    def __str__(self) -> str:
        return self.__repr__()
