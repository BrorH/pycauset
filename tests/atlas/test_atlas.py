"""The operation atlas as a test (review/features/ATLAS.md §4).

One test per operation runs every cell of that operation (dtype x structure x view x storage,
plus the dtype-pair and structure-pair cells) and fails on

  * a wrong result (✗) that is not in KNOWN_GAPS (each entry names its finding);
  * an atlas harness error (a bug in the oracle must be fixed, not ignored);
  * a regression: a cell that was correct in the committed snapshot (review/atlas/atlas.json.gz)
    and now raises or is wrong.

Paths (fast / generic / numpy) are reported by `python -m tests.atlas.report`, not asserted
here: they are performance facts, worked down in Phase N / C.5.
"""
from __future__ import annotations

import gzip
import json
import os

import pytest

from tests.atlas.atlas import OPS, enumerate_cells, run_cell

# cell key -> finding id. A fixed gap shows up as a failure here ("listed but correct"):
# remove the entry.
KNOWN_GAPS: dict[str, str] = {}

_REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "review", "atlas", "atlas.json.gz")


def _previously_ok() -> set[str]:
    try:
        with gzip.open(_REPORT, "rt") as f:
            snapshot = json.load(f)
    except (OSError, ValueError):
        return set()
    return {key for key, value in snapshot.items() if value.startswith("ok")}


_CELLS = enumerate_cells(storages=("ram", "file"))
_OK_BEFORE = _previously_ok()


@pytest.mark.parametrize("op", [o.name for o in OPS])
def test_operation_atlas(op):
    problems = []
    for cell in (c for c in _CELLS if c.op == op):
        run_cell(cell)
        key = cell.key()
        gap = KNOWN_GAPS.get(key)
        if cell.outcome == "harness":
            problems.append(f"{key}: atlas harness error {cell.error}")
        elif cell.outcome == "wrong" and not gap:
            problems.append(f"{key}: WRONG {cell.error}")
        elif gap and cell.outcome == "ok":
            problems.append(f"{key}: listed as gap {gap} but correct now; remove it from KNOWN_GAPS")
        elif key in _OK_BEFORE and cell.outcome in ("unsupported", "raises", "wrong"):
            problems.append(f"{key}: regression ({cell.outcome}: {cell.error})")
    assert not problems, f"{len(problems)} atlas problems:\n" + "\n".join(problems[:40])
