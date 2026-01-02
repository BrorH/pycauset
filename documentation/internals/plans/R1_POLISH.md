# R1_POLISH: Professionalism & Quality Assurance

**Goal:** Ensure `pycauset` meets high professional standards (NumPy-like quality) before Release 1. This involves cleaning up packaging, enforcing code style, and standardizing documentation.

## 1. Packaging Hygiene ("DLL Hell" Prevention)
- [ ] **Move DLLs:** Stop dumping loose DLLs (`cublas64_12.dll`, etc.) in the root `python/pycauset` folder.
- [ ] **Create `libs` directory:** Move runtime binaries to `python/pycauset/libs`.
- [ ] **Runtime Hook:** Update `__init__.py` to call `os.add_dll_directory()` for the `libs` folder on Windows startup.
- [ ] **Wheel Audit:** Ensure wheels are self-contained and don't conflict with other CUDA-using libraries.

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
