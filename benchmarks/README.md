# benchmarks/

Speed scripts; the numbers and how to read them are in `BENCHMARKS.md` (repository root).
Per-operation timing against NumPy is `python -m tests.atlas.timing`, not a script here.
Scripts here are run by hand (CI runs `parity.py` report-only); a script that stops
running is fixed or deleted, not left behind.
