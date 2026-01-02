from __future__ import annotations

import atexit
import os
import weakref
from pathlib import Path
from typing import Any, Callable


class Runtime:
    def __init__(
        self,
        *,
        cleanup_storage: Callable[[Path], None],
        set_temporary_file: Callable[[Path, bool], None],
        env_var: str = "PYCAUSET_STORAGE_DIR",
    ) -> None:
        self._cleanup_storage = cleanup_storage
        self._set_temporary_file = set_temporary_file
        self._env_var = env_var
        self._storage_root_cache: Path | None = None
        self._storage_roots_seen: set[Path] = set()
        self._live_matrices: weakref.WeakSet = weakref.WeakSet()

    def storage_root(self) -> Path:
        if self._storage_root_cache is not None:
            self._storage_roots_seen.add(self._storage_root_cache)
            return self._storage_root_cache

        base = Path.cwd().resolve() / ".pycauset"

        base.mkdir(parents=True, exist_ok=True)
        self._storage_root_cache = base
        self._storage_roots_seen.add(base)
        return base

    def set_storage_root(self, root: Path) -> Path:
        """Set the directory used for auto-created backing files.

                Notes:
                - This affects Python's cleanup/temporary tracking.
                - Switching the root mid-session can leave temp files in the old root;
                    we mitigate this by tracking all roots used and cleaning them on exit.
        """
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._storage_root_cache = root
        self._storage_roots_seen.add(root)
        # Remove stale temp files from previous runs in the new root.
        self._cleanup_storage(root)
        return root

    def has_live_matrices(self) -> bool:
        try:
            return len(self._live_matrices) > 0
        except TypeError:
            return True

    def initial_cleanup(self) -> None:
        self._cleanup_storage(self.storage_root())

    def track_matrix(self, instance: Any) -> None:
        try:
            self._live_matrices.add(instance)
        except TypeError:
            pass

    def release_tracked_matrices(self) -> None:
        for matrix in list(self._live_matrices):
            close = getattr(matrix, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def register_cleanup(self, *, keep_temp_files_getter: Callable[[], bool]) -> None:
        def _finalize() -> None:
            try:
                self.cleanup_all_roots(keep_temp_files=keep_temp_files_getter())
            except Exception:
                pass

        atexit.register(_finalize)

    def cleanup_all_roots(self, *, keep_temp_files: bool) -> None:
        """Release live matrices and clean all known storage roots."""

        self.release_tracked_matrices()
        if keep_temp_files:
            return

        roots = set(self._storage_roots_seen)
        try:
            roots.add(self.storage_root())
        except Exception:
            pass

        for root in roots:
            try:
                self._cleanup_storage(root)
            except Exception:
                pass

    def mark_temporary_if_auto(self, matrix: Any) -> None:
        if not hasattr(matrix, "get_backing_file"):
            return

        try:
            path = Path(matrix.get_backing_file()).resolve()
            root = self.storage_root().resolve()

            # Path.is_relative_to is 3.9+; we support 3.8+ historically? repo is 3.12 now.
            if path.is_relative_to(root):
                if hasattr(matrix, "set_temporary"):
                    matrix.set_temporary(True)
                else:
                    self._set_temporary_file(path, True)
        except (ValueError, OSError, AttributeError):
            pass
