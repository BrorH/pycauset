# Algorithms

The algorithms behind the operations, as they run in R1 (CPU only; see
[[internals/Compute Architecture|Compute Architecture]] for the dispatch and the file map).

## Matrix multiplication

- **float32 / float64 / complex:** BLAS `?gemm` (OpenBLAS), called directly for in-memory
  operands; out-of-core operands are tiled by the [[internals/Streaming Manager|Streaming Manager]].
- **Integers:** exact through float64 `dgemm` when the measured operand bounds guarantee
  every partial sum is an integer below 2^53 (F-055); otherwise integer loops parallelised
  with `ParallelFor`. Overflow of the result dtype raises.
- **Bit matrices** (`bool`): each entry of `C @ C` is a popcount of `row_i AND column_j` over
  64-bit words: AVX-512 `VPOPCNTDQ` when the CPU has it (checked at run time), else
  `std::popcount`. The result counts paths (int32), as NumPy's integer matmul does.
- **Structured operands:** identity and diagonal products are O(N²) scalings; triangular
  products skip the zero triangle and keep the class when the result has it.

## Eigenvalues

- **Dense:** LAPACK `?syevd` / `?heevd` for symmetric and Hermitian input, `?geev`
  otherwise, in the operand's precision.
- **`eigvals_arnoldi(a, k, m, tol)`:** the top-k eigenvalues by magnitude through Spectra's
  restarted Krylov solvers (Lanczos for symmetric input, Arnoldi otherwise). It needs only
  matrix-vector products, so `a` may be file-backed. Non-convergence raises (no partial
  answers); problems too small for the Krylov parameters use the dense solver.
- **Skew matrices:** `eigvals_skew` / `eig_skew` use the real-antisymmetric structure
  (purely imaginary pairs).

## Inverse and solves

`invert` is NumPy's algorithm: `solve(A, I)` by LU with partial pivoting (`?gesv`). An
upper triangular matrix's inverse is upper triangular (the class is kept), a diagonal's is
elementwise, an identity's is `1/scale`. `solve_triangular` checks that `A` is triangular,
diagonal or identity by class and then runs `solve`.

## Structure-aware shortcuts

Structure is the class, so a shortcut can never disagree with the values: identity and
diagonal matmul / inverse / determinant are O(N) or O(1), a triangular determinant is the
product of its diagonal, Hermitian and symmetric eigen problems use `?heevd` / `?syevd`. Each
operation's entry function lists its structure rules
([[project/protocols/Adding Operations|Adding Operations]] §3).

## Sprinkling

`pc.causet(n, spacetime=MinkowskiDiamond(d), seed)` (`src/causet/Sprinkler.cpp`):

1. Points are drawn in blocks of 10,000; block `b` has its own generator seeded by a hash
   of `(seed, b)` (SplitMix64), so the points depend only on `(spacetime, n, seed)`.
   Each point is drawn uniformly in the diamond by rejection from its bounding box.
2. The points are sorted by time (stable), which labels the elements so that every
   relation lands in the upper triangle.
3. For every pair `i < j` the spacetime's causality test (Minkowski: `dt > 0` and
   `dt² > |dx|²`) sets bit `(i, j)` of a `TriangularMatrix<bool>`.

The `CausalSet` keeps only the seed: `coordinates()` regenerates the points (all `n` of
them, then picks the requested indices). Sprinkling holds all `n·d` coordinates in RAM while
it runs and the pair loop is serial O(n²); both are on the speed list (review PLAN Phase 6).

## Derivation: Retarded Propagator ($K_R$)

We need to calculate the Retarded Propagator matrix $K_R$ for massive causal sets ($N \approx 10^6$). The generalized definition for a scalar field on a causal set is:

$$ K_R = \Phi(I - b\Phi)^{-1} $$

Where:
*   $\Phi = a C$
*   $C$: $N \times N$ Causal Matrix (Strictly Upper Triangular, Binary).
*   $I$: Identity Matrix.
*   $a, b$: Scalar constants derived from the spacetime dimension $d$, sprinkling density $\rho$, and field mass $m$.

### Derivation

Direct inversion of $(I - b\Phi)$ is $O(N^3)$ and produces a dense matrix. We can transform this into a form solvable by our existing efficient kernel.

Substitute $\Phi = aC$:
$$ K_R = aC(I - abC)^{-1} $$

Let $\alpha_{eff} = -\frac{1}{ab}$. Then:
$$ K_R = -\frac{1}{b} \left[ C (\alpha_{eff}I + C)^{-1} \right] $$

The term in the brackets, $X = C(\alpha_{eff}I + C)^{-1}$, is exactly the form solved by our existing `compute_k` kernel (which solves $X = C(a_{kernel}I+C)^{-1}$).

### How `compute_k` computes it

$X (\alpha I + C) = C$ with $C$ strictly upper triangular gives each column of $X$ by back
substitution, $X_{ij} = (C_{ij} - \sum_{i<m<j} C_{im} X_{mj}) / \alpha$, where the sum runs only
over the set bits of row $i$ of $C$ (`std::countr_zero` on the packed words). Columns are
independent and run in parallel (`ParallelFor`), each needing one column of workspace. The
result is a `TriangularMatrix` (`CpuSolver::compute_k_matrix`).
