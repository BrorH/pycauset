"""List C++ source functions that never executed, from a gcovr JSON report.

Template instantiations of the same source function are merged by (file, line):
a function counts as executed if ANY instantiation ran. Output groups by file.

Produce the report (see BASELINE.md "C++ coverage"):
  gcovr -r . build/cov --filter src/ --filter include/ --exclude src/accelerators/ \
        --json cppcov.json
Run:
  python review/scripts/cpp_unexecuted.py cppcov.json [--min-lines N]
"""
from __future__ import annotations

import collections
import json
import sys


def main() -> None:
    data = json.load(open(sys.argv[1]))
    min_lines = int(sys.argv[sys.argv.index("--min-lines") + 1]) if "--min-lines" in sys.argv else 1
    funcs: dict[tuple[str, int], dict] = {}
    for f in data["files"]:
        for fn in f.get("functions", []):
            key = (f["file"], fn["lineno"])
            name = fn.get("demangled_name") or fn["name"]
            short = name.split("(")[0]
            ent = funcs.setdefault(key, {"name": short, "count": 0})
            ent["count"] += fn.get("execution_count", 0)
        # crude function length: lines until next function start in same file
    starts = collections.defaultdict(list)
    for (file, line) in funcs:
        starts[file].append(line)
    dead = collections.defaultdict(list)
    total_dead = 0
    for (file, line), ent in funcs.items():
        if ent["count"]:
            continue
        nxt = [l for l in starts[file] if l > line]
        length = (min(nxt) - line) if nxt else 0
        if length and length < min_lines:
            continue
        dead[file].append((line, ent["name"], length))
        total_dead += 1
    n_all = len(funcs)
    print(f"{total_dead} of {n_all} source functions never executed\n")
    for file in sorted(dead, key=lambda k: -len(dead[k])):
        print(f"## {file} ({len(dead[file])})")
        for line, name, length in sorted(dead[file]):
            print(f"   L{line:<5} {name[-90:]}" + (f"  (~{length} lines)" if length else ""))


if __name__ == "__main__":
    main()
