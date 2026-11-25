# Operation atlas: every operation × dtype × structure × view, correct and fast or not

**Asked by:** maintainer, 2026-11-25. "We keep discovering the same class of issue: operation
X with dtype A isn't optimized compared to Y with B (or is wrong). Catalogue them all."
**Status:** designed (session 7); PLAN Phase C. Replaces hunting by probe.

## 1. Why the same bug class keeps appearing
Each operation is implemented as a ladder of special cases (per dtype, per structure, per
view flag), and the fallback at the bottom of each ladder differs: sometimes a correct but
slow virtual per-element loop, sometimes a NumPy round trip, sometimes code that silently
ignores a view flag. Sessions 5–7 found these one at a time (F-001, F-043, F-049–F-063): the
probes looked where someone thought to look. The atlas looks everywhere, mechanically, and
keeps looking (it runs in CI).

## 2. Axes (the cells)
| axis | values |
|---|---|
| operation | every public operation: element-wise `+ - * /` (matrix⊕matrix, ⊕scalar, reflected), unary `-`, `~`, bitwise `& \| ^`, comparisons; reductions (`sum`, `trace`, `norm`, `dot`); products (`@` matrix/vector/vector-matrix, outer); linear algebra (`inv`, `solve`, `det`, `eig*`, `svd*`, `qr`, `lu`, `cholesky`, `pinv`, `matrix_power`, `lstsq`); construction and conversion (`matrix`, `vector`, `zeros/ones/empty`, `astype`/cast, export to NumPy); indexing (get, set, slice, slice-assign); persistence (save/load round trip); causal-set kernels (links, transitive closure/reduction, interval counts) |
| dtype | the 14 dtypes (bool/bit, int8–64, uint8–64, float16/32/64, complex64/128, complex32) |
| structure | dense, triangular (strict and with diagonal), diagonal, identity, symmetric, antisymmetric, C ± Cᵀ closure, dense vector, unit vector |
| view | plain, transposed, scaled (int and float factors), conjugated, Hermitian, window at (0,0), window with offset, lazy (closure, integer factor) |
| storage | RAM, file-backed (forced via the memory threshold) |
| second operand | for binary operations: same dtype/structure, plus every dtype pair at dense/plain (dtype promotion), plus every structure pair at float64 |

Full products are large; the enumeration is mechanical but pruned by rules that are
themselves recorded (e.g. "linear algebra: floats and complex; integers are written out to
float64 first, which the atlas verifies once per operation").

## 3. What is measured per cell
1. **Correctness** against an oracle: NumPy on the same logical values (dtype and values;
   exact for integers, tolerance tiers per LAZY.md for floats), and the same exception type
   where NumPy raises. Unsupported cells must raise a clear error, never return a result.
2. **Path**, i.e. how the result was computed, classified automatically:
   * `fast`: a typed kernel (BLAS/LAPACK, word-level bit kernel, contiguous typed loop);
   * `written-out`: the input was copied to plain storage first, then a fast kernel (fine for
     O(n³) operations; flagged for O(n²) ones);
   * `generic`: one virtual element accessor call per element (correct, slow);
   * `numpy-fallback`: the Python layer called NumPy;
   * `unsupported`: raises.
   Detection is objective, not hand-tagged: (a) the kernel trace (`debug_trace` names,
   completed so every kernel entry sets one); (b) an accessor counter: a thread-local counter
   in the virtual element accessors, enabled only by a debug flag, so "≥ n² accessor calls"
   identifies `generic` without anyone having to know; (c) a Python-side hook on the NumPy
   fallback sites.
3. **Speed** for the cells that matter (dense-equivalent sizes, 1024² and 1M elements):
   time vs NumPy on the same operation, and vs the same operation on float64 dense (so
   "int16 matmul is 50× slower than float64" is a visible number, as in F-055).

## 4. Outputs
* `review/atlas/atlas.json`: every cell with result, path, timing.
* `review/atlas/ATLAS.md`: generated tables per operation (rows dtype × structure, columns
  views), each cell a symbol (✓ fast, ◐ written-out, ○ generic, N numpy, ✗ wrong, — raises),
  plus the worst ratios. This is the "what is and isn't optimized" answer, always current.
* CI: the correctness part runs as tests (`tests/atlas/`); known gaps are listed explicitly
  (strict xfail with a finding ID), so a fixed gap must be removed from the list and a new
  gap fails the build.
* Replaces `python/pycauset/_internal/support_matrix.py` + `tools/check_support_matrix.py`
  (a hand-maintained, partial version of the same idea) and the ad-hoc probes in
  `review/probes/`.

## 5. Policy the atlas enforces
* No ✗ cells, ever (correctness first).
* No `numpy-fallback` in the engine (R1 "NumPy for causal sets": NumPy is an oracle, not a
  backend), except documented interop entry points.
* `generic` is allowed only for small, documented cases (e.g. float16 arithmetic); every
  other `generic` cell is a finding with a planned fix.
* An operation is "optimized for dtype A" when its time at the reference size is within 2× of
  the best dtype of the same width class, or a finding explains why not.

## 6. Build order
1. Instrumentation: complete kernel traces; accessor counter (debug flag); fallback hook.
2. Enumerator + oracle + classifier, correctness only (fast to run); first full run →
   findings for every ✗ cell (expected: many in the less-travelled structures and dtypes).
3. Timing pass and the generated report.
4. CI wiring, delete `support_matrix.py`, `tools/check_support_matrix.py`, superseded probes.
5. Work the report down (Phase 4/6 items reference atlas cells instead of ad-hoc findings).
