# Lazy evaluation

Some operations return their result without computing every element. The result is an
ordinary matrix of the usual class and dtype; the elements are computed when they are read,
usually inside another operation that reads them anyway. You never have to change your code:
laziness only changes speed and memory.

## What is lazy

| expression | cost | stored |
|---|---|---|
| `k * A`, `A / k`, `-A` (float, complex, and integer dense/triangular/symmetric matrices) | O(1) | the factor `k`; `A`'s storage is shared |
| `A.T`, `A.H`, `A.conj()`, slices `A[i:j, k:l]` | O(1) | a view of `A`'s storage |
| `C + C.T`, `C - C.T` for a causal matrix `C` | O(1) | nothing: computed from `C`'s bits |
| `A + A.T`, `A - A.T` (any matrix) | one pass | the packed symmetric / antisymmetric result (half the memory) |

`A.is_lazy` says whether elements are computed when read: a factor, a transpose, conjugation, or
a closure such as `C + C.T`. A dense slice with step 1 shares `A`'s storage (O(1)), but its
elements are the stored ones, read in place, so `A[0:2, 0:2].is_lazy` is False.

Chains fold: `2 * (3 * A)` stores one factor, 6. Matrix-vector products on `C - C.T` read the
bits directly, so iterative eigensolvers (`eigvals_arnoldi`) never build the full matrix.

## Guarantees

* **Same values and type as computing immediately.** Integer results are exact and wrap like
  NumPy's (`int8`: `100 * 3 == 44`).
* **Floating point:** a lazy factor is applied in double precision and chains fold into one
  factor, so results can differ from NumPy's step-by-step evaluation:
  * in the **last digit** (one ulp) for float16 / float32 / complex64 data: `A * 0.1` scales
    by the double 0.1 and rounds once, NumPy rounds 0.1 to float32 first (about a fifth of
    the elements differ by one ulp);
  * where an intermediate step would overflow or underflow: `(A * 1e10) * 1e-20` is
    `A * 1e-10`, so 1e300 gives 1e290 where NumPy's first step gives inf.

  Results are deterministic: the same program gives the same bits on every run and machine.
  For NumPy's exact steps, `pc.materialize` the intermediate result.
* **No aliasing.** Writing into `A` never changes an earlier `B = 2 * A` or `A.T` (unlike
  NumPy, where `a.T` shares memory with `a`). Writing into a lazy result is allowed; it is
  stored at that point.
* **Saving** stores the computed result, so files never depend on other files.

## Inspecting and controlling it

```python
import pycauset as pc

C = pc.causal_matrix(100, seed=1)
D = C - C.T
assert D.is_lazy and D.dtype == "int32"   # computed from C's bits on read
E = pc.materialize(D)                    # same values, now stored
assert not E.is_lazy and E[3, 7] == D[3, 7]
```

`pycauset.materialize(A)` is only needed to trade memory for repeated-access speed (e.g.
before millions of single-element reads).
