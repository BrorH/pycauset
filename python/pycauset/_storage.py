import os
from pathlib import Path

def cleanup_storage(storage_root: Path) -> None:
    """Scan storage directory and delete temporary files (.tmp)."""
    if not storage_root.exists():
        return
        
    for item in storage_root.iterdir():
        if not item.is_file():
            continue
            
        if item.name.endswith(".tmp") or item.name.endswith(".raw_tmp"):
            try:
                item.unlink()
            except OSError:
                pass 

def set_temporary_file(path: Path, is_temp: bool) -> None:
    """
    Mark a file as temporary.

    In the current storage architecture, temporary files are identified by their
    filename extension (e.g. `.tmp`, `.raw_tmp`). This function is intentionally
    a no-op because renaming an in-use backing file is not reliable across
    platforms.
    """
    return

def is_temporary_file(path: Path) -> bool:
    """Check if a file is temporary based on extension."""
    return path.name.endswith(".tmp") or path.name.endswith(".raw_tmp")

class StorageRegistry:
    """
    Registry for storage management.
    """
    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root

    def register_auto_file(self, path: str) -> None:
        pass

    def finalize(self, keep_files: bool) -> None:
        if not keep_files:
            cleanup_storage(self._root)
