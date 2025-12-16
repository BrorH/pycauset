from __future__ import annotations

import os
from pathlib import Path
import importlib.machinery
import importlib.util
import sys
from importlib import import_module
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

    # Always add the package directory first.
    dirs: list[str] = []
    try:
        dirs.append(str(Path(package_dir).resolve()))
    except Exception:
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
    except Exception:
        # Best-effort only.
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
    # Prefer loading the extension module that lives alongside this Python package.
    # This avoids accidentally importing a different installed copy (e.g. from site-packages)
    # when developing from a source checkout.
    if module.startswith("."):
        try:
            pkg = import_module(package)
            pkg_dir = next(iter(getattr(pkg, "__path__", [])), None)
        except Exception:
            pkg_dir = None

        if pkg_dir:
            mod_name = module.lstrip(".")
            full_name = f"{package}.{mod_name}"

            existing = sys.modules.get(full_name)
            if existing is not None:
                return existing

            search_dirs: list[str] = []
            try:
                pkg_path = Path(pkg_dir).resolve()
                repo_root = pkg_path.parents[1]
                build_root = repo_root / "build"
                for d in (
                    build_root / "Release",
                    build_root / "Debug",
                    build_root / "RelWithDebInfo",
                    build_root / "MinSizeRel",
                    build_root,
                    pkg_path,
                ):
                    if d.exists() and d.is_dir():
                        search_dirs.append(str(d))
            except Exception:
                search_dirs.append(pkg_dir)

            # Prefer build outputs (when present) to keep the extension module and
            # the shared core DLL in sync during source checkout development.
            for base in search_dirs:
                for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                    candidate = os.path.join(base, mod_name + suffix)
                    if not os.path.exists(candidate):
                        continue
                    spec = importlib.util.spec_from_file_location(full_name, candidate)
                    if spec is None or spec.loader is None:
                        continue
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod

    return import_module(module, package=package)


def safe_get(native: Any, name: str) -> Any:
    return getattr(native, name, None)
