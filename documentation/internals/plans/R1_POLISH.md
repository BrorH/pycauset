# R1_POLISH: Professionalism & Quality Assurance

**Goal:** Ensure `pycauset` meets high professional standards (NumPy-like quality) before Release 1. This involves cleaning up packaging, enforcing code style, and standardizing documentation.

## 1. Packaging Hygiene ("DLL Hell" Prevention)
- [x] **Move DLLs:** runtime binaries (`pycauset_core.dll`, `libopenblas.dll`) no longer live in `python/pycauset/`; the stray `bin/` install and the Windows import `.lib` were also removed.
- [x] **Create `libs` directory:** CMake installs runtime DLLs to `python/pycauset/libs`.
- [x] **Runtime Hook:** `configure_windows_dll_search_paths()` (in `_internal/native.py`) adds `pycauset/libs` via `os.add_dll_directory()`, with the package dir as a backwards-compatible fallback for source checkouts.
- [x] **Wheel Audit:** `python -m build --wheel` verified — wheel installs into a fresh venv and `import pycauset` + `matmul` work (DLLs resolved from `libs/`).

## 2. Documentation Standards
- [ ] **Fix Links:** Convert all Obsidian-style `[[wiki_links]]` to standard Markdown `[Link](path.md)` syntax.
    - *Target:* `documentation/index.md` and other doc files.
- [ ] **Render Check:** Ensure documentation builds correctly with `mkdocs` and renders correctly on GitHub/PyPI.

## 3. Code Quality & Linting
- [ ] **Configure Ruff:** Add `[tool.ruff]` to `pyproject.toml`.
    - Enforce NumPy-style docstrings (Rule `D`).
    - Enforce modern Python idioms (Rule `UP`).
    - Enforce import sorting (Rule `I`).
- [ ] **Configure MyPy:** Add `[tool.mypy]` to `pyproject.toml` for static type checking.
- [ ] **Baseline:** Run linters and fix immediate low-hanging fruit (unused imports, undefined variables).

## 4. Build System Cleanup
- [ ] **Audit CMake:** Review `CMakeLists.txt` for aggressive warning suppressions (e.g., `/wd4251`, `/wd4996`).
- [ ] **Fix Warnings:** Address the underlying C++ issues instead of silencing the compiler where possible.

## 5. Namespace Refactoring
- [ ] **Slim `__init__.py`:** The main `python/pycauset/__init__.py` is too large (~1800 lines).
- [ ] **Move Logic:** Extract implementation details to `_internal` modules.
- [ ] **Public API:** Ensure `__init__.py` only exposes the intended public API.

## 6. Codebase Cleanup
- [ ] **Dead Code Removal:** Remove legacy "eager" evaluation paths in `MatrixBase` once `R1_LAZY` is stable.
- [ ] **Temporary File Logic:** Remove obsolete manual temporary file creation logic that is superseded by the `MemoryGovernor` spill mechanism.
