"""Every Python block of the listed user guides runs, top to bottom, as one script
(TESTS.md layout: docs snippets). A guide joins the list once the docs truth pass (PLAN
Phase 5) has rewritten it against the code."""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKED = ["README.md", "documentation/guides/Quickstart.md", "documentation/guides/Examples.md",
           "documentation/guides/User Guide.md"]


@pytest.mark.parametrize("page", CHECKED)
def test_guide_code_runs(page, tmp_path, monkeypatch):
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / page).read_text(), re.S)
    assert blocks, f"{page} has no python blocks"
    monkeypatch.chdir(tmp_path)  # files the guide writes land in a temporary directory
    import pycauset as pc

    try:
        exec(compile("\n".join(blocks), page, "exec"), {"__name__": "__doc_snippet__"})
    finally:
        pc.debug_mode = False  # a guide may switch it on
