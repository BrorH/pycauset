"""Probe: execute every ```python block in the user-facing docs.

numpy/pycauset are pre-imported as np, pc and pycauset. Blocks that do not parse
(signature listings) are skipped and counted. Blocks within a page run in order in one namespace (so later blocks can use earlier
names). Each page runs in its own subprocess with a timeout. Plotting is made
headless (fig.show() is a no-op). A page "fails" at its first failing block.

Run:  python review/probes/run_doc_snippets.py [path ...]   (default: guides/ + docs/ + README)

Each file is capped at 1 GiB (RLIMIT_FSIZE) and the page's working dir is removed
afterwards: some doc examples allocate very large matrices.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)

RUNNER = r'''
import atexit, json, os, resource, shutil, sys, tempfile, traceback, warnings
warnings.simplefilter("ignore")
# Cap any single file at 1 GiB so a doc example cannot fill the disk.
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 30, 1 << 30))
_work = tempfile.mkdtemp(prefix="docsnip_")
atexit.register(shutil.rmtree, _work, True)
os.chdir(_work)
try:
    import plotly.basefigure as _bf
    _bf.BaseFigure.show = lambda self, *a, **k: None
except Exception:
    pass
blocks = json.load(open(sys.argv[1]))
ns = {"__name__": "__doc__"}
exec("import numpy as np\nimport pycauset\nimport pycauset as pc", ns)
skipped = 0
for i, src in enumerate(blocks):
    try:
        code = compile(src, f"<block {i}>", "exec")
    except SyntaxError:
        skipped += 1  # signature-style block, not meant to run
        continue
    try:
        exec(code, ns)
    except Exception as e:
        line = traceback.format_exc().strip().splitlines()[-1]
        print(json.dumps({"block": i, "error": line[:240], "src": src.strip().splitlines()[0][:100] if src.strip() else ""}))
        sys.exit(1)
print(json.dumps({"ok": len(blocks) - skipped, "skipped": skipped}))
'''


def pages(args):
    if args:
        yield from (pathlib.Path(a).resolve() for a in args)
        return
    yield ROOT / "README.md"
    for sub in ("documentation/guides", "documentation/docs"):
        yield from sorted((ROOT / sub).rglob("*.md"))


def main() -> None:
    runner = pathlib.Path(tempfile.mkdtemp()) / "runner.py"
    runner.write_text(RUNNER)
    n_ok = n_fail = n_none = 0
    for page in pages(sys.argv[1:]):
        blocks = BLOCK.findall(page.read_text(errors="ignore"))
        if not blocks:
            n_none += 1
            continue
        bf = runner.with_name("blocks.json")
        bf.write_text(json.dumps(blocks))
        try:
            proc = subprocess.run([sys.executable, str(runner), str(bf)], capture_output=True,
                                  text=True, timeout=120, env={**__import__("os").environ})
            out = (proc.stdout.strip().splitlines() or ["{}"])[-1]
            res = json.loads(out) if out.startswith("{") else {"error": (proc.stderr.strip().splitlines() or ["crash"])[-1]}
        except subprocess.TimeoutExpired:
            res = {"error": "TIMEOUT (120s)"}
        for d in pathlib.Path(tempfile.gettempdir()).glob("docsnip_*"):
            __import__("shutil").rmtree(d, ignore_errors=True)
        rel = page.relative_to(ROOT)
        if "ok" in res:
            n_ok += 1
        else:
            n_fail += 1
            print(f"FAIL {rel} [block {res.get('block', '?')}/{len(blocks)}]: {res.get('error')}")
            if res.get("src"):
                print(f"     first line: {res['src']}")
    print(f"\npages ok={n_ok} fail={n_fail} (no python blocks: {n_none})")


if __name__ == "__main__":
    main()
