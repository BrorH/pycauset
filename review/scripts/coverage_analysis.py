"""Analyse a per-test coverage database (pytest --cov-context=test).

Two reports:
  functions  - Python functions/methods with ZERO executed lines across the whole run
               (dead-but-referenced or untested code; candidates for review/removal)
  tests      - per test FILE: lines covered, lines covered by no other file ("unique"),
               and runtime-free redundancy hint (unique == 0 means fully overlapped)

Produce the database:
  python -m pytest tests/python --cov=python/pycauset --cov-context=test --cov-report=
Run:
  python review/scripts/coverage_analysis.py .coverage functions
  python review/scripts/coverage_analysis.py .coverage tests
"""
from __future__ import annotations

import ast
import collections
import pathlib
import sys

import coverage


def functions_report(cov: coverage.Coverage) -> None:
    data = cov.get_data()
    rows = []
    for fname in sorted(data.measured_files()):
        executed = set(data.lines(fname) or [])
        src = pathlib.Path(fname).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_lines = {n.lineno for n in ast.walk(node) if hasattr(n, "lineno")} - {node.lineno}
                # ignore decorators/docstring-only
                if body_lines and not (body_lines & executed):
                    rows.append((fname.split("pycauset/", 1)[-1], node.lineno, node.name,
                                 (node.end_lineno or node.lineno) - node.lineno + 1))
    by_file = collections.Counter(r[0] for r in rows)
    total = sum(r[3] for r in rows)
    print(f"{len(rows)} never-executed functions, ~{total} lines\n")
    for f, n in by_file.most_common():
        print(f"## {f}  ({n})")
        for r in rows:
            if r[0] == f:
                print(f"   L{r[1]:<5} {r[2]}  ({r[3]} lines)")


def tests_report(cov: coverage.Coverage) -> None:
    data = cov.get_data()
    per_file: dict[str, set] = collections.defaultdict(set)
    for fname in data.measured_files():
        for line, ctxs in (data.contexts_by_lineno(fname) or {}).items():
            for c in ctxs:
                if not c:
                    continue
                tfile = c.split("::")[0].split("/")[-1].split("|")[0]
                per_file[tfile].add((fname, line))
    owner = collections.Counter()
    for s in per_file.values():
        owner.update(s)
    rows = []
    for t, s in per_file.items():
        unique = sum(1 for x in s if owner[x] == 1)
        rows.append((unique, len(s), t))
    rows.sort()
    print(f"{'unique':>6} {'covered':>7}  test file")
    for u, n, t in rows:
        print(f"{u:>6} {n:>7}  {t}")
    print(f"\nfiles with zero unique lines: {sum(1 for r in rows if r[0] == 0)} / {len(rows)}")


if __name__ == "__main__":
    cov = coverage.Coverage(data_file=sys.argv[1])
    cov.load()
    {"functions": functions_report, "tests": tests_report}[sys.argv[2]](cov)
