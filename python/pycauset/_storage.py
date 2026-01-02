import os
from pathlib import Path
from typing import Iterable, List, Set

_TEMP_TRACKER: Set[Path] = set()


def _normalize(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_temporary_name(name: str) -> bool:
    return name.endswith(".tmp") or name.endswith(".raw_tmp")


def is_temporary_file(path: Path) -> bool:
    """Check if a file is temporary based on extension."""
    try:
        return _is_temporary_name(path.name)
    except Exception:
        return False


def _prune_tracker() -> None:
    for entry in list(_TEMP_TRACKER):
        try:
            if not entry.exists():
                _TEMP_TRACKER.discard(entry)
        except Exception:
            _TEMP_TRACKER.discard(entry)


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def record_temporary_file(path: Path) -> None:
    """Track a temporary file path for later cleanup/diagnostics."""
    try:
        normalized = _normalize(path)
    except Exception:
        return
    _TEMP_TRACKER.add(normalized)


def tracked_temp_files(root: Path | None = None) -> List[Path]:
    """Return known temporary files (optionally filtered by root)."""
    _prune_tracker()
    if root is None:
        return sorted(_TEMP_TRACKER)
    normalized_root = _normalize(root)
    return sorted(p for p in _TEMP_TRACKER if _is_under_root(p, normalized_root))


def clear_tracked_temp_files(root: Path | None = None) -> None:
    """Forget tracked temp files (optionally scoped to a root)."""
    _prune_tracker()
    if root is None:
        _TEMP_TRACKER.clear()
        return
    normalized_root = _normalize(root)
    for p in list(_TEMP_TRACKER):
        if _is_under_root(p, normalized_root):
            _TEMP_TRACKER.discard(p)


def _iter_temp_files(storage_root: Path) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(storage_root):
        for name in filenames:
            path = Path(dirpath) / name
            if is_temporary_file(path) or path in _TEMP_TRACKER:
                yield path


def cleanup_storage(storage_root: Path) -> None:
    """Scan storage directory and delete temporary files (.tmp/.raw_tmp)."""
    if not storage_root.exists():
        return

    root = _normalize(storage_root)
    for temp_path in list(_iter_temp_files(root)):
        try:
            temp_path.unlink()
        except OSError:
            # Best effort: ignore paths that cannot be removed (e.g., locked).
            pass

    clear_tracked_temp_files(root)


def set_temporary_file(path: Path, is_temp: bool) -> None:
    """Mark a file as temporary for lifecycle tracking.

    Renaming in-use backing files is unreliable on many platforms, so we treat
    this as a bookkeeping hook instead of attempting to mutate the path.
    """
    if not is_temp:
        return
    try:
        record_temporary_file(path)
    except Exception:
        return


class StorageRegistry:
    """Registry for storage management and cleanup."""

    def __init__(self, storage_root: Path) -> None:
        self._root = _normalize(storage_root)

    def register_auto_file(self, path: str) -> None:
        try:
            candidate = _normalize(Path(path))
        except Exception:
            return
        if not _is_under_root(candidate, self._root):
            return
        if not is_temporary_file(candidate):
            return
        record_temporary_file(candidate)

    def finalize(self, keep_files: bool) -> None:
        if not keep_files:
            cleanup_storage(self._root)
        else:
            _prune_tracker()
