# Findings: everyday-use scenarios

This document records what happened when we exercised PyCauset through
realistic "everyday" causal-set workflows, each implemented at least two
different ways and cross-checked against NumPy. It lists the bugs we found and
fixed, the remaining limitations, and benchmark numbers (including a NumPy
comparison and a large-N check).

Scenarios live in `tests/python/test_everyday_scenarios.py`. Benchmarks live in
`benchmarks/scenario_benchmarks.py` (NumPy comparison) and
`benchmarks/large_n_check.py` (large-N correctness and timing).

## Scenarios and results

| Scenario | Ways exercised | Result |
|---|---|---|
| Cube 3 causal matrices (3D diamond) and add them | `(C@C)@C` chain, `matrix_power(C, 3)` | Both agree with NumPy |
| Count 2-step causal relations (`C^2`) | `@`, `pc.matmul`, `pc.dot` | All three agree with NumPy |
| Count elements in each element's causal future (row sums) | `C @ ones`, NumPy row sums | Agree |
| Constant working matrix without a dtype | lazy `ones` vs explicit `bool_`, lazy `zeros` then `fill` | Agree |
| `K = C (I + C)^-1` | `pc.compute_k` vs NumPy `inv` | Agree (rtol 1e-10) |

## Bugs found and fixed

1. **`TriangularIntegerMatrix` could not be exported to NumPy.**
   `to_numpy(C @ C)` raised `TypeError: data type 'triangularinteger' not
   understood`. Fixed by adding the `triangularinteger -> int32` entry to the
   export guard's type-name-to-dtype map.

2. **Cubing a causal matrix threw "Unsupported matrix multiplication types".**
   `(C @ C) @ C` failed because `C @ C` yields a `TriangularIntegerMatrix`, and
   `TriangularIntegerMatrix x {TriangularBitMatrix, TriangularIntegerMatrix}` had
   no native dispatch. Fixed with new C++ cases and a correctness-first kernel.

3. **Dense mixed-dtype matmul threw for common combinations.** `int32 @ float64`,
   `float64 @ int32`, `int32 @ TriangularBitMatrix`, `float64 @ TriangularBitMatrix`
   (and their reverses) all raised `Unsupported matrix multiplication types`.
   Fixed by adding a general fallback at the end of the C++ dispatch that resolves
   the promoted result dtype via `PromotionResolver` and computes through a generic
   element-access path (with an `int32` result branch in the CPU solver).

4. **`matrix_power(C, 4)` densified to a dense `IntegerMatrix`.** The
   `TriangularIntegerMatrix` operands were being caught by the "same-dtype integer"
   case (which matches on `get_data_type()`), producing a dense result. Fixed by
   moving the triangular-int case ahead of the same-dtype integer case; powers now
   stay `TriangularIntegerMatrix`.

5. **`pc.dot(vector, matrix)` returned a row vector `(1, n)` instead of NumPy's
   1-D `(n,)`.** Fixed by un-transposing the native vec-mat result inside `pc.dot`.

6. **Wide mixed-integer matmul threw.** `int64 @ int32`, `int8 @ int16`,
   `uint8 @ uint16`, `uint32 @ int32`, and other mixed-width integer combinations
   raised `Unsupported matrix multiplication types`. Fixed by extending the general
   fallback to every integer width (int8/int16/int32/int64/uint8/uint16/uint32/
   uint64) via a templated generic integer kernel.

7. **Naive correctness-first kernels were unacceptably slow.** The
   `DenseBitMatrix x TriangularBitMatrix` matmul and the causal-matrix matvec
   (`C @ vector`) used naive element-access loops. Added a popcount kernel for
   bit x bit (materializing a triangular bit operand to dense first) and
   scale-first matvec kernels for `TriangularBitMatrix x {int32, float64, bool}`
   vectors. See benchmarks for the speedups.

## Remaining limitations

1. **int64/uint64 mixed matmul is exact only up to 2^53.** The generic mixed-kind
   integer path accumulates in `double` (exact for every other integer width). For
   `int64`/`uint64` results the values are exact only below 2^53; beyond that the
   result can lose precision. Same-dtype `int64 @ int64` / `uint64 @ uint64` use
   the dedicated exact kernels and are unaffected.

2. **`pc.empty(shape)` without a dtype raises on first use.** Intentional ("no
   silent wrong answers"): an empty allocation has no dtype until a value is
   written, so reading it raises a clear `TypeError`. Callers must `fill(...)` or
   `set(...)` first, or pass `dtype=`.

## Benchmarks (NumPy comparison)

Informal wall-clock, best of 5, `n = 256` (3D diamond), Windows MSVC. The NumPy
reference uses dense `int64` arrays except the float64 row (BLAS-backed). `B @ C`
and the matvec numbers reflect the optimized popcount / scale-first kernels.

| Operation | pycauset (ms) | numpy (ms) | ratio |
|---|---|---|---|
| `C @ C` (triangular bit x bit) | 0.090 | 24.97 | 0.00x |
| `B @ C` (dense bool x triangular bit) | 0.717 | 23.31 | 0.03x |
| `(C @ C) @ C` (cube) | 2.73 | 40.36 | 0.07x |
| `matrix_power(C, 3)` | 8.07 | 44.02 | 0.18x |
| `dot(C, C)` | 0.71 | 20.60 | 0.03x |
| `C @ ones` (matvec) | 0.046 | 0.032 | 1.44x |
| float64 dense matmul (BLAS) | 0.52 | 0.19 | 2.74x |

Notes:

- For integer matmul, pycauset is much faster because its bit-packed popcount
  kernel is `O(N^3/64)` while NumPy's integer matmul is a naive (non-BLAS) loop.
  `B @ C` dropped from ~10.8 ms to ~0.7 ms (about 15x) after adding the popcount
  path for mixed bit operands.
- `C @ ones` matvec went from a generic `O(N^2)` element loop to a scale-first
  `O(set bits)` kernel. At `n = 256` it is ~1.4x NumPy; at larger N the sparse
  triangular form wins decisively (see below).
- For float64 dense matmul, NumPy's BLAS wins by ~2.7x (0.19 ms vs 0.52 ms); this
  remains the main gap and is part of the post-R1 float64-parity program.

## Large-N check

`benchmarks/large_n_check.py` verifies `C @ C` correctness on a random sample of
entries (direct `O(N)` reference) plus a structural invariant (strictly upper
triangular square has zero diagonal). All checks passed up to N = 80000.

| N | build (s) | `C @ C` (s) | correct |
|---|---|---|---|
| 2000 | 0.016 | 0.006 | yes |
| 5000 | 0.094 | 0.059 | yes |
| 10000 | 0.368 | 0.409 | yes |
| 20000 | 1.465 | 3.234 | yes |
| 40000 | 5.834 | 24.700 | yes |
| 80000 | 23.315 | 181.676 | yes |

At N = 40000 and N = 80000 the out-of-core memory governor kicked in and spilled
the intermediate result to disk ("RAM full ... falling back to disk", ~12 GB at
N = 80000); the computation still completed correctly. This exercises the
disk-backed path end to end at scale.

## Matvec at larger N

The scale-first triangular matvec is sparse-aware, so it beats NumPy's dense
`int64` matvec once N grows:

| N | pycauset (ms) | numpy (ms) | ratio |
|---|---|---|---|
| 256 | 0.048 | 0.032 | 1.49x |
| 1000 | 0.053 | 0.454 | 0.12x |
| 2000 | 0.108 | 2.433 | 0.04x |
| 5000 | 0.483 | 16.03 | 0.03x |

## Remaining performance work (post-R1)

The targeted correctness-first optimizations are done. The remaining performance
gaps are the full post-R1 program: float64 matmul parity with BLAS (currently
~2.7x behind), GPU parity, and streaming-everything.
