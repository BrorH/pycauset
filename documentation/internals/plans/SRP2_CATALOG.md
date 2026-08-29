# SRP-2, Causal Math Optimization Catalog (R2_CATALOG)

Maps the operator combinations causal-set theory actually uses to the numerical
shortcuts they admit. Each entry is a *routing target*: the shortcut must be
implemented (or explicitly routed/blocked) so the field core never falls back to a
general O(n³) path when structure makes it O(n²) or better.

## 1. Propagator `K_R = aC(I − baC)⁻¹`

- **Structure:** `C` is a strictly-upper-triangular bit matrix, so `M = I − baC` is
  upper-triangular with unit diagonal. `K_R` is upper-triangular.
- **Shortcut:** solve `M X = aC` by **triangular back-substitution** (`solve_triangular`),
  not a general LU/`inv`. `K_R = K_Aᵀ` for the advanced propagator.
- **Massless limit:** `b → 0` gives `K_R = aC` (no solve at all).

## 2. Pauli–Jordan `iΔ = K_R − K_A`

- **Structure:** `Δ` is real **antisymmetric**; `iΔ` is Hermitian.
- **Shortcut:** store `Δ` as an `AntiSymmetricMatrix` (half the memory) with a scalar
  `1j`, rather than a dense complex matrix. (Already the R1 `ScalarField.pauli_jordan`
  representation.)

## 3. Sorkin–Johnston `W = positive part of iΔ`

- **Structure:** `iΔ` is Hermitian; equivalently `Δ` is real antisymmetric (eigenvalues
  pure-imaginary, ±iσ).
- **Shortcut:** diagonalize the **real antisymmetric** `Δ` with a skew-symmetric
  eigensolver (`eigvals_skew`) rather than a general complex `eigh`. `W` needs the
  eigenvectors, so pair `eigvals_skew` with a skew eigensystem if available, else fall
  back to `eigh`. (Current implementation uses `np.linalg.eigh`; a native skew path is
  the optimization.)

## 4. Causal structure / Hasse

- **Structure:** `C` is a bit matrix; the link (Hasse) matrix is `L = C & ~(C²)`
  (transitive reduction).
- **Shortcut:** bit-matrix popcount matmul (`C @ C` over bits) instead of dense integer
  matmul, then a bitwise `& ~`. (The engine's `DenseBitMatrix × TriangularBitMatrix`
  popcount kernel already targets this.)

## 5. Structural / property shortcuts (already gospel)

- `matrix_rank`, `det`, `trace`, `norm`, `invert` on `Identity`/`Diagonal`/`Triangular`/
  `Zero` matrices use closed forms (properties-as-gospel), skipping SVD/LU.

## Routing status

| shortcut | implemented | notes |
| :-- | :-: | :-- |
| triangular solve for `K_R` | ✅ | `CpuSolver::compute_k_matrix` does column-wise back-substitution over the bit matrix (O(n²), no `inv`); pinned against the naive dense `inv` by `test_retarded_matches_formula` |
| antisymmetric `iΔ` storage | ✅ | `AntiSymmetricMatrix` + scalar `1j` |
| skew eigensolver for `W` | ✅ | native `eig_skew` returns eigenvalues + eigenvectors (LAPACK `dgeev` + top-k sort, pinned by `test_skew.py`). `wightman()` still routes through `np.linalg.eigh` because the general `dgeev` path is not yet faster than `dsyevd`; the routing flips to `eig_skew` once a dedicated skew tridiagonalization lands |
| bit-popcount `C @ C` for links | ✅ | engine popcount kernel |

All four shortcuts are now implemented. The `W` routing note above is the one
follow-up left in the field core: `eig_skew` exists, but `wightman()` keeps using
`eigh` until the skew tridiagonalization makes `eig_skew` the faster path.
