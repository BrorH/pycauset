import os
import re
from pathlib import Path
from typing import Iterable

def _normalize(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_temporary_name(name: str) -> bool:
    return name.endswith(".tmp") or name.endswith(".raw_tmp")


def is_temporary_file(path: Path) -> bool:
    """Check if a file is temporary based on extension."""
    return _is_temporary_name(path.name)


def _iter_temp_files(storage_root: Path) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(storage_root):
        for name in filenames:
            path = Path(dirpath) / name
            if is_temporary_file(path):
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
