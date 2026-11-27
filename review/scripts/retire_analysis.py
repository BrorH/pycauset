"""Which old test files (tests/python) cover Python lines the new suite does not?

TESTS.md section 4, rule 3: an old file may be deleted when every line it covers is covered
by the new suite (tests/engine, tests/atlas) or is confirmed dead. Run:

  COVERAGE_FILE=py.coverage PYTHONPATH=python python -m pytest tests/engine tests/python \
      tests/atlas --cov=python/pycauset --cov-context=test --cov-report=
  python review/scripts/retire_analysis.py py.coverage [--lines FILE]

Prints, per old file, the number of Python lines only old files cover ("old-only") and how
many of those only this file covers. C++ lines: review/scripts/cpp_retire_analysis.py.
"""
from __future__ import annotations

import collections
import sys

import coverage

NEW = ("tests/engine/", "tests/atlas/", "tests/physics/")
OLD = "tests/python/"


def main() -> None:
    cov = coverage.Coverage(data_file=sys.argv[1])
    cov.load()
    data = cov.get_data()
    new_lines: set = set()
    old_by_file: dict[str, set] = collections.defaultdict(set)
    for fname in data.measured_files():
        short = fname.split("python/pycauset/", 1)[-1]
        for line, ctxs in (data.contexts_by_lineno(fname) or {}).items():
            for c in ctxs:
                if c.startswith(NEW):
                    new_lines.add((short, line))
                elif c.startswith(OLD):
                    old_by_file[c.split("::")[0]].add((short, line))
    old_only = {f: s - new_lines for f, s in old_by_file.items()}
    owners = collections.Counter(x for s in old_only.values() for x in s)
    rows = sorted((len(s), sum(1 for x in s if owners[x] == 1), f) for f, s in old_only.items())
    print(f"{'old-only':>8} {'sole':>5}  file")
    for n, sole, f in rows:
        print(f"{n:>8} {sole:>5}  {f}")
    zero = [f for n, _, f in rows if n == 0]
    print(f"\n{len(zero)} of {len(rows)} old files cover no Python line the new suite misses")
    if "--lines" in sys.argv:
        target = sys.argv[sys.argv.index("--lines") + 1]
        for short, line in sorted(old_only.get(target, ())):
            print(f"LINE {short}:{line}")


if __name__ == "__main__":
    main()
