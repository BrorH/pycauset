"""Print the per-file inventory tables used in review/INVENTORY.md.

Re-run after moving/adding files and merge the output into INVENTORY.md by hand,
keeping the existing Review/Notes columns. Lines = physical lines (wc -l).

Run:  python review/scripts/inventory_table.py > /tmp/inv.md
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

GROUPS = [
    ("C++ headers: core", "include/pycauset/core", "*.hpp"),
    ("C++ headers: matrix / vector / expression", "include/pycauset/matrix", "**/*.hpp"),
    ("", "include/pycauset/vector", "*.hpp"),
    ("C++ headers: compute / math / causet", "include/pycauset/compute", "**/*.hpp"),
    ("", "include/pycauset/math", "*.hpp"),
    ("", "include/pycauset/causet", "*.hpp"),
    ("C++ sources", "src", "**/*.cpp"),
    ("C++ binding helpers", "src/bindings", "*.hpp"),
    ("CUDA (on hold)", "src/accelerators/cuda", "*"),
    ("Python package", "python/pycauset", "*.py"),
    ("", "python/pycauset", "*.pyi"),
    ("Python internals", "python/pycauset/_internal", "*.py"),
]


def main() -> None:
    seen: set[pathlib.Path] = set()
    for title, d, pat in GROUPS:
        if title:
            print(f"\n### {title}\n\n| File | Lines | Review | Notes |\n|---|---:|:-:|---|")
        for p in sorted((ROOT / d).glob(pat)):
            if not p.is_file() or p in seen or "cuda" in p.parts and "CUDA" not in title:
                continue
            seen.add(p)
            n = sum(1 for _ in p.open(errors="ignore"))
            print(f"| `{p.relative_to(ROOT)}` | {n} | ☐ | |")


if __name__ == "__main__":
    main()
