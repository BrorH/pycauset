"""R2_QA: verify every public symbol in `pycauset.__all__` has a doc page.

Checks each name against the documentation tree (`documentation/docs/**`). A name
is "documented" when a page `pycauset.<name>.md` exists, or when the name is a
re-export of a submodule class documented under its dotted namespace
(`pycauset.field.Foo`, `pycauset.spacetime.Foo`, `pycauset.cuda.foo`).

Exit code 0 means zero undocumented public symbols; exit code 1 prints the gaps.
"""

from __future__ import annotations

import glob
import os
import sys

import pycauset

DOCS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documentation", "docs")


def main() -> int:
    all_doc_tokens = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(DOCS_ROOT, "**", "*.md"), recursive=True)
    }
    # Dotted submodule namespace pages (pycauset.field.X / spacetime.X / cuda.x).
    dotted = {
        t
        for t in all_doc_tokens
        if t.startswith("pycauset.")
        and any(f".{p}." in t or t.endswith(f".{p}") for p in ("cuda", "spacetime", "field"))
    }

    missing = []
    for name in sorted(pycauset.__all__):
        if f"pycauset.{name}" in all_doc_tokens:
            continue
        if name in ("cuda", "spacetime", "synthetic", "field"):
            continue
        if any(d.endswith("." + name) for d in dotted):
            continue
        missing.append(name)

    if missing:
        print(f"Undocumented public symbols ({len(missing)}):")
        for name in missing:
            print("  -", name)
        return 1
    print(f"OK: all {len(pycauset.__all__)} public symbols are documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
