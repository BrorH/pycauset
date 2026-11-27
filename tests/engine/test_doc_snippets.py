"""The Python code of every documentation page runs (PLAN 5.1): README.md, the guides and the
API pages. A page's blocks run top to bottom as one script, starting at the first block
that imports something; blocks before it are signatures (an API page opens with one)."""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_PAGES = [ROOT / "README.md", *sorted((ROOT / "documentation" / "guides").glob("*.md")),
          *sorted((ROOT / "documentation" / "docs").rglob("*.md"))]


def _code(page: pathlib.Path) -> list[str]:
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    first = next((i for i, b in enumerate(blocks) if re.search(r"^\s*(import|from) ", b, re.M)), None)
    return [] if first is None else blocks[first:]


CHECKED = [str(p.relative_to(ROOT)) for p in _PAGES if _code(p)]


@pytest.mark.parametrize("page", CHECKED)
def test_guide_code_runs(page, tmp_path, monkeypatch):
    blocks = _code(ROOT / page)
    monkeypatch.chdir(tmp_path)  # files the guide writes land in a temporary directory
    try:  # guides call fig.show(); no browser in tests
        import plotly.basedatatypes

        monkeypatch.setattr(plotly.basedatatypes.BaseFigure, "show", lambda self, *a, **k: None)
    except ImportError:
        pass
    import warnings

    import pycauset as pc

    root, threshold, lazy = pc._storage_root(), pc.get_memory_threshold(), pc._native._get_lazy()
    pc._native._set_lazy(True)  # the pages describe the default (PYCAUSET_LAZY=0 runs too)
    try:
        exec(compile("\n".join(blocks), page, "exec"), {"__name__": "__doc_snippet__"})
    finally:  # guides change global settings; restore them
        pc._native._set_lazy(lazy)
        pc.debug_mode = False
        pc.set_memory_threshold(threshold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pc.set_backing_dir(root)
