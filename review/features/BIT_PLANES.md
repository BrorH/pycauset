# Investigation: arbitrary-width integer dtypes (2-, 3-, 4-bit, ...)

**Asked by:** maintainer, 2026-11-25. "If we have two bool matrices that add together, going
all the way to int8 is informational overkill. Would this require too big a rewrite,
especially with existing eigensolvers?"
**Status:** investigated; recommendation below (no new user-facing dtypes; an internal
"bit-plane" storage encoding later, behind value bounds).

## 1. What the workloads actually need (measured, n = 3000 sprinkled causet)
| object | value range | bits needed | handled today by |
|---|---|---|---|
| C, links L, C + Cᵀ | [0, 1] | 1 | bit storage / closure view (0 extra) |
| C − Cᵀ | [−1, 1] | 2 | closure view over C's bits (0 extra) |
| C + L, small sums of bit matrices | [0, 2] | 2 | int32 today (int8 with value bounds) |
| L @ L (two-link paths) | [0, 6] | 3 | int32 |
| C @ C (interval sizes) | [0, 2892] | 12 | int32 (int16 with value bounds) |
| propagators, Pauli–Jordan, W | real | float | float64 |

So the "between 1 and 8 bits" objects are real but narrow: small integer combinations of bit
matrices (BD-operator-like sums, few-link path counts). The headline objects are either
1 bit (already optimal), need ≥ 12 bits (int16 is within 1.3× of optimal), or are floats.

## 2. What a k-bit integer would cost and gain (measured, n = 2000)
The natural representation is **bit planes**: a k-bit matrix stored as k bit matrices
(plane p holds bit p of every element). Arithmetic then reuses the existing bit kernels:
* addition = a ripple-carry adder on 64-bit words (bit + bit: low plane `A ^ B`, carry `A & B`);
* matmul = Σ_{p,q} 2^{p+q} (A_p @ B_q): k² popcount bit-matmuls;
* matvec (all iterative eigensolvers) = Σ_p 2^p (A_p x): k bit-matvecs.

| operation | int8 / int16 (today) | bit planes (estimated from bit kernels) |
|---|---|---|
| element-wise add, 2-bit result | int8 2.8 ms | XOR + AND = 0.8 ms (3.5×) |
| matmul, 2-bit × 2-bit | int16 5.9 s | 4 × 0.049 s = 0.2 s (30×) |
| matmul reference | float64 BLAS 0.11 s | — |
| storage, 2-bit values | int8: 8 bits | 2 bits (4×) |

Two side findings from the measurements:
* **F-054 (fixed):** bit `&`/`|`/`^` had lost their word-level path in the structure-class
  rewrite (71 ms → 0.4 ms after the fix).
* **F-055:** integer matmul has no fast path at all (int16 5.9 s vs float64 BLAS 0.11 s). With
  value bounds (F1) integer matmul can run *exactly* through float64 BLAS whenever the
  bounds prove every partial sum < 2^53. That is a 50× win for every integer matmul and
  needs no new dtype. **This is the bigger lever than new widths.**

## 3. Eigensolvers and the numerical ecosystem
* **Dense eigensolvers (LAPACK)** only take float32/float64 (and complex). Every integer
  dtype, existing or new, is converted to float64 first (`to_memory_flat_real_square`). A
  2-bit storage would save storage, not eigensolver time or memory.
* **Iterative eigensolvers (Arnoldi/Lanczos via Spectra)** only need matvecs. Bit planes plug
  in directly (k bit-matvecs per product), exactly like the C ± Cᵀ closure already does.
* **NumPy/SciPy/BLAS** have no 2–7-bit types: export must widen (to int8), and every library
  call converts. That is the "100 years of infrastructure" cost: real, but confined to the
  boundary if the width is an internal encoding rather than a user dtype.

## 4. Cost of doing it as new user-facing dtypes (`int2`, `int3`, `uint4`, ...)
Large and permanent. Each new dtype enters the dtype × dtype promotion table (16 → 20+
dtypes = 400+ pairs), needs NumPy-interop rules for a type NumPy lacks, a persistence
encoding, a class instantiation per structure, and every kernel path (element-wise,
scalar, matmul, matvec, reductions, export, streaming). It also changes user-visible
semantics (what does `int3 + int3` return? NumPy has no answer to copy). It's the
feature-creep risk you flagged earlier.

## 5. Recommendation
1. **No new user-facing dtypes.** Users keep NumPy's 2^N dtypes; `dtype=` stays the manual
   override.
2. **F-055 first (biggest, simplest win):** exact integer matmul through float64 BLAS when
   value bounds guarantee no partial sum exceeds 2^53. Belongs with value bounds (F1).
3. **Bit-plane storage as an internal encoding, later, behind value bounds:** a matrix whose
   *guaranteed* bounds fit in ≤ 4 bits may be stored as bit planes while reporting its
   normal logical dtype (e.g. int8): the same "logical dtype ≠ storage" mechanism as the
   closure view (LAZY.md L2b). Laziness rules apply: invisible except for speed/memory.
   Size: similar to BitClosureMatrix plus an adder and the k² matmul; not a rewrite.
   Build it only when a real workload needs it (e.g. a BD-operator pipeline on small
   integer sums); today the 1-bit and ≥ 12-bit objects dominate.
