import os
import re
from pathlib import Path
from typing import Iterable, Set

_TEMP_TRACKER: Set[Path] = set()


def _normalize(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_temporary_name(name: str) -> bool:
    return name.endswith(".tmp") or name.endswith(".raw_tmp")


def is_temporary_file(path: Path) -> bool:
    """Check if a file is temporary based on extension."""
    return _is_temporary_name(path.name)


def _prune_tracker() -> None:
    for entry in list(_TEMP_TRACKER):
        try:
            exists = entry.exists()
        except OSError:  # unreadable parent: treat as gone
            exists = False
        if not exists:
            _TEMP_TRACKER.discard(entry)


def _is_under_root(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def record_temporary_file(path: Path) -> None:
    """Track a temporary file path for later cleanup/diagnostics."""
    _TEMP_TRACKER.add(_normalize(path))


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


_PID_IN_NAME = re.compile(r"_(\d+)_\d+\.(?:raw_)?tmp$")


def _owned_by_live_other_process(path: Path) -> bool:
    """Temp files are named <prefix>_<pid>_<n>.tmp (StorageUtils.cpp). A file of another
    process that is still running is not ours to delete (several processes may share one
    storage directory). On Windows a file in use cannot be deleted anyway."""
    m = _PID_IN_NAME.search(path.name)
    if m is None or os.name == "nt":
        return False
    pid = int(m.group(1))
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    except OSError:
        return False
    return True


def cleanup_storage(storage_root: Path) -> None:
    """Scan storage directory and delete temporary files (.tmp/.raw_tmp)."""
    if not storage_root.exists():
        return

    root = _normalize(storage_root)
    for temp_path in list(_iter_temp_files(root)):
        if _owned_by_live_other_process(temp_path):
            continue  # another process sharing the directory is still using it
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
    record_temporary_file(path)


