import os
import struct
from pathlib import Path
from typing import Set

# Header offset for is_temporary is 56 (after scalar)
# struct FileHeader {
#     char magic[8];          // 0
#     uint32_t version;       // 8
#     uint32_t matrix_type;   // 12
#     uint32_t data_type;     // 16
#     uint8_t padding[4];     // 20
#     uint64_t rows;          // 24
#     uint64_t cols;          // 32
#     uint64_t seed;          // 40
#     double scalar;          // 48
#     uint8_t is_temporary;   // 56
#     ...
# };

_HEADER_SIZE = 4096
_IS_TEMP_OFFSET = 56

def is_temporary_file(path: Path) -> bool:
    """Check if a pycauset file is marked as temporary in its header."""
    try:
        if not path.exists() or path.stat().st_size < _HEADER_SIZE:
            return False
        with path.open("rb") as f:
            f.seek(_IS_TEMP_OFFSET)
            byte = f.read(1)
            if not byte:
                return False
            return byte[0] != 0
    except OSError:
        return False

def set_temporary_file(path: Path, is_temp: bool) -> None:
    """Update the temporary flag in the file header."""
    try:
        if not path.exists():
            return
        with path.open("r+b") as f:
            f.seek(_IS_TEMP_OFFSET)
            f.write(b'\x01' if is_temp else b'\x00')
    except OSError:
        pass

def cleanup_storage(storage_root: Path) -> None:
    """Scan storage directory and delete files marked as temporary."""
    if not storage_root.exists():
        return
        
    for item in storage_root.iterdir():
        if not item.is_file() or not item.name.endswith(".pycauset"):
            continue
            
        if is_temporary_file(item):
            try:
                item.unlink()
            except OSError:
                pass # File might be locked or in use

class StorageRegistry:
    """
    Legacy registry kept for compatibility, but now delegates to header-based cleanup.
    """
    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root

    def register_auto_file(self, path: str) -> None:
        # No-op: we now set the bit on the matrix object or file directly
        # However, if the file is created but not yet opened as a matrix object,
        # we might need to set the bit here.
        # But register_auto_file is called when we resolve the path.
        # The file might not exist yet.
        # So we can't set the bit here.
        # We must rely on the matrix constructor or method to set it.
        pass

    def finalize(self, keep_files: bool) -> None:
        if not keep_files:
            cleanup_storage(self._root)
