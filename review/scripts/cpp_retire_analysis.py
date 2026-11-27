"""C++ lines an old test batch executes that the new suite does not (TESTS.md section 4).

Two gcovr JSON reports from the coverage build (build/cov), counters reset in between
(delete the *.gcda files):

  find build/cov -name '*.gcda' -delete
  PYTHONPATH=<cov package> pytest tests/engine tests/atlas
  gcovr -r . build/cov --filter src/ --filter include/ --json new.json
  find build/cov -name '*.gcda' -delete
  PYTHONPATH=<cov package> pytest <old files>
  gcovr -r . build/cov --filter src/ --filter include/ --json old.json
  python review/scripts/cpp_retire_analysis.py new.json old.json

Prints the old-only lines grouped by file (empty: the batch adds no C++ coverage).
"""
from __future__ import annotations

import collections
import json
import sys


def executed(path: str) -> set[tuple[str, int]]:
    data = json.load(open(path))
    out = set()
    for f in data["files"]:
        for ln in f.get("lines", []):
            if ln.get("count", 0) > 0 and not ln.get("gcovr/noncode", False):
                out.add((f["file"], ln["line_number"]))
    return out


def main() -> None:
    new, old = executed(sys.argv[1]), executed(sys.argv[2])
    extra = old - new
    by_file = collections.defaultdict(list)
    for f, n in sorted(extra):
        by_file[f].append(n)
    print(f"{len(extra)} C++ lines executed only by the old batch ({len(old)} executed by it)")
    for f, lines in sorted(by_file.items(), key=lambda x: -len(x[1])):
        print(f"  {len(lines):5}  {f}  {lines[:12]}{' ...' if len(lines) > 12 else ''}")


if __name__ == "__main__":
    main()
