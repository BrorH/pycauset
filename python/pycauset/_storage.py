import json
import os
from pathlib import Path
from typing import Iterable, Set

try:  # pragma: no cover - platform specific
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - platform specific
    msvcrt = None

try:  # pragma: no cover - platform specific
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - platform specific
    fcntl = None


class _FileLock:
    """Minimal cross-platform file lock using fcntl/msvcrt."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a+b")
        if msvcrt is not None:  # pragma: no cover - windows only
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        elif fcntl is not None:  # pragma: no cover - posix only
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is None:
            return False
        if msvcrt is not None:  # pragma: no cover - windows only
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif fcntl is not None:  # pragma: no cover - posix only
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        self._handle.close()
        self._handle = None
        return False


class StorageRegistry:
    """Tracks auto-generated backing files to clean them up between runs."""

    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root
        self._manifest_path = self._root / "manifest.json"
        self._lock_path = self._root / "manifest.lock"
        self._session_paths: Set[str] = set()

    def register_auto_file(self, path: str) -> None:
        normalized = str(Path(path).resolve())
        self._session_paths.add(normalized)

    def finalize(self, keep_files: bool) -> None:
        try:
            self._flush(keep_files)
        except Exception:
            pass

    # Internal helpers -------------------------------------------------
    def _flush(self, keep_files: bool) -> None:
        if not self._session_paths and not self._manifest_path.exists():
            return
        with _FileLock(self._lock_path):
            existing = self._read_manifest_locked()
            all_paths = existing | self._session_paths
            if keep_files:
                self._write_manifest_locked(all_paths)
            else:
                self._delete_files(all_paths)
                self._write_manifest_locked(set())
        self._session_paths.clear()

    def _read_manifest_locked(self) -> Set[str]:
        if not self._manifest_path.exists():
            return set()
        try:
            with self._manifest_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return set()
        entries = payload.get("auto_files", [])
        if not isinstance(entries, list):
            return set()
        normalized = set()
        for entry in entries:
            if isinstance(entry, str):
                normalized.add(str(Path(entry)))
        return normalized

    def _write_manifest_locked(self, paths: Set[str]) -> None:
        if not paths:
            try:
                self._manifest_path.unlink()
            except FileNotFoundError:
                pass
            return
        temp_path = self._manifest_path.with_suffix(".json.tmp")
        data = {"auto_files": sorted(paths)}
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        temp_path.replace(self._manifest_path)

    def _delete_files(self, paths: Iterable[str]) -> None:
        for entry in paths:
            try:
                os.remove(entry)
            except FileNotFoundError:
                continue
            except OSError:
                continue
