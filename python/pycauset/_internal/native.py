from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any


def configure_windows_dll_search_paths(*, package_dir: str) -> None:
    """Best-effort DLL search path setup for Windows.

    This keeps import-time behavior in one place.
    """

    if os.name != "nt":
        return

    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is None:
        return

    # Always add the package directory first, plus its `libs/` subdir (where the
    # bundled Windows DLLs live, pycauset_core.dll + libopenblas.dll).
    dirs: list[str] = []
    try:
        pkg = Path(package_dir).resolve()
        dirs.append(str(pkg / "libs"))
        dirs.append(str(pkg))
    except OSError:
        dirs.append(package_dir)

    # In a source checkout, native builds often live in ./build/<config>/.
    # If the extension module was copied into the package directory but the
    # shared core DLL was not, we must add the build output dir so Windows can
    # resolve `pycauset_core.dll` consistently.
    try:
        repo_root = Path(package_dir).resolve().parents[1]
        build_root = repo_root / "build"
        for candidate in (
            build_root,
            build_root / "Release",
            build_root / "Debug",
            build_root / "RelWithDebInfo",
            build_root / "MinSizeRel",
        ):
            if candidate.exists() and candidate.is_dir():
                dirs.append(str(candidate))
    except OSError:  # an unreadable parent directory: no extra search path
        pass

    seen: set[str] = set()
    for d in dirs:
        if not d or d in seen:
            continue
        seen.add(d)
        try:
            add_dir(d)
        except OSError:
            pass


def import_native_extension(*, package: str, module: str = "._pycauset") -> Any:
    """The extension beside this package's __init__.py (a regular package's only path entry).
    It used to search build directories and load the newest `_pycauset*` by modification time, a
    guess that decided which binary ran (review 3.8 C-22); developers copy the build output into
    the package (CLAUDE.md), and its RUNPATH `$ORIGIN` loads the core library beside it."""
    return import_module(module, package=package)
