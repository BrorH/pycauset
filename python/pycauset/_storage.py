import os
import re
from pathlib import Path
from typing import Iterable

def _normalize(path: Path) -> Path:
    return Path(path).expanduser().resolve()


# The engine names its temporary backing files <prefix>_<pid>_<n>.pycauset-tmp
# (StorageUtils.cpp). The sweep deletes only names of exactly that form, directly in the storage
# root: it used to delete every *.tmp / *.raw_tmp file below the root, recursively, whoever made
# it (review 3.8 C-1).
TEMP_SUFFIX = ".pycauset-tmp"
_TEMP_NAME = re.compile(r"^[A-Za-z0-9_]*?_(\d+)_\d+" + re.escape(TEMP_SUFFIX) + "$")


def is_temporary_file(path: Path) -> bool:
    """Whether `path` is a temporary backing file of the engine (by its name)."""
    return _TEMP_NAME.match(Path(path).name) is not None


def _iter_temp_files(storage_root: Path) -> Iterable[Path]:
    try:
        entries = list(os.scandir(storage_root))
    except OSError:
        return
    for entry in entries:
        if _TEMP_NAME.match(entry.name) is not None and entry.is_file(follow_symlinks=False):
            yield Path(entry.path)


def _owned_by_live_other_process(path: Path) -> bool:
    """A temp file of another process that is still running is not ours to delete (several
    processes may share one storage directory). On Windows a file in use cannot be deleted
    anyway."""
    m = _TEMP_NAME.match(path.name)
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
    """Delete the engine's temporary files directly in `storage_root` whose process is gone,
    and this process's own. Nothing else is touched."""
    if not storage_root.exists():
        return

    root = _normalize(storage_root)
    for temp_path in list(_iter_temp_files(root)):
        if _owned_by_live_other_process(temp_path):
            continue  # another process sharing the directory is still using it
        try:
            temp_path.unlink()
        except OSError:
            # Best effort: a file in use (Windows) or already gone stays for the next sweep.
            pass
