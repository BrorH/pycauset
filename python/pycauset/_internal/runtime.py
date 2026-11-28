from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Callable


class Runtime:
    def __init__(
        self,
        *,
        cleanup_storage: Callable[[Path], None],
        env_var: str = "PYCAUSET_STORAGE_DIR",
    ) -> None:
        self._cleanup_storage = cleanup_storage
        self._env_var = env_var
        self._storage_root_cache: Path | None = None
        self._storage_roots_seen: set[Path] = set()

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
        """Set the directory used for auto-created backing files. Every root used is swept at
        exit (objects made before the switch keep their files in the old one)."""
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._storage_root_cache = root
        self._storage_roots_seen.add(root)
        # Remove stale temp files from previous runs in the new root.
        self._cleanup_storage(root)
        return root

    def has_own_temp_files(self, root: Path) -> bool:
        """Whether objects of this process are file-backed in `root` (temp names carry the
        pid: `<prefix>_<pid>_<n>.tmp`)."""
        marker = f"_{os.getpid()}_"
        try:
            return any(marker in p.name for p in root.glob("*.tmp"))
        except OSError:
            return False

    def initial_cleanup(self) -> None:
        self._cleanup_storage(self.storage_root())

    def register_cleanup(self) -> None:
        def _finalize() -> None:
            # At exit there is no one to report to: a directory that cannot be read or a
            # file that cannot be deleted stays (the next import sweeps again).
            try:
                self.cleanup_all_roots()
            except OSError:
                pass

        atexit.register(_finalize)

    def cleanup_all_roots(self) -> None:
        """Sweep every storage root this process used: its own temporary files (objects still
        alive at exit; POSIX deletes a mapped file, Windows keeps it and the next import's
        sweep gets it) and leftovers of crashed runs. Files of other live processes stay."""

        roots = set(self._storage_roots_seen)
        roots.add(self.storage_root())
        for root in roots:
            self._cleanup_storage(root)  # best effort per file (OSError handled inside)

