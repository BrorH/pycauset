# tests/

NumPy is the oracle: every test compares with what NumPy gives on the same data. Strategy:
`review/TESTS.md`; how to run: `documentation/dev/Testing & Benchmarks.md`.

| Path | What |
|---|---|
| `engine/` | the engine suite by subject; `test_regressions.py` holds one `test_F<id>` per fixed bug (ids in `review/FINDINGS.md`); `test_doc_snippets.py` runs the code of every documentation page |
| `atlas/` | every operation x dtype x structure x view x storage against NumPy (`pytest tests/atlas` is the CI gate); `report.py` and `timing.py` write `review/atlas/` |
| `blockmatrix/` | the Python-level block matrices |
| `*.cpp` | C++ unit tests (gtest; `-DBUILD_TESTS=ON`, then `ctest`) |
