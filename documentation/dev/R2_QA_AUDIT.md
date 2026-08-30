# R2_QA, public-surface audit (partial)

Audit of the R2 public API surface and per-node test/doc coverage. This is the
"in-step docs + tests" gate (R2_QA); it does not write docs/tests, it verifies.

## Public surface (all resolve)

- `spacetime`: `Spacetime`, `register`, `create`, `export_python`, `get_registry`,
  `RestrictedSpacetime`, `TransformedSpacetime`, `ConformalSpacetime`,
  `PeriodicSpacetime`, `MinkowskiDiamond/Cylinder/Box`, `DeSitter`, `AntiDeSitter`,
  `FLRW`, `Schwarzschild`.
- `field`: `Field`, `CorrelatedField`, `ContinuumCorrelatedField`, `ScalarField`,
  callable `pc.field("scalar", …)`.
- `CausalSet` methods: `validate`, `links`, `past`, `future`, `interval`, `is_chain`,
  `is_antichain`, `longest_chain`, `layers`, `relation_fraction`,
  `myrheim_meyer_dimension`, `plot_embedding`, `plot_hasse`, `plot_causal_matrix`.
- `synthetic`: `chain`, `antichain`, `transitive_percolation`, `random_dag_order`,
  `product_order`, `poset`.
- Top-level: `pc.plot_embedding/hasse/causal_matrix`, `pc.show`.

## Test coverage (full suite: 812 passed + 23 skipped; the R2 node files are listed below)

| node | test file |
| :-- | :-- |
| R2_SIG / R2_ABC | `test_spacetime_r2_abc.py` |
| R2_VALIDATE / R2_EMBED | `test_causet_r2_sprinkle_validate.py` |
| R2_MINK | `test_minkowski_r2.py` |
| R2_FIELD / R2_KRD / R2_SJ | `test_field_r2.py` |
| R2_CONV / R2_CMVP | `test_conv_continuum_r2.py` |
| R2_CREATE | `test_create_r2.py` (18) |
| R2_VIZ | `test_viz_r2.py` (11) |
| R2_STRUCT | `test_struct_r2.py` |
| R2_SYNTH | `test_synth_r2.py` |
| R2_DIM | `test_dim_r2.py` |
| R2_BATCH | `test_batch_r2.py` |
| R2_CURVED / R2_BH | `test_curved_r2.py`, `test_bh_r2.py` |
| R2_ENT | `test_ent_r2.py` |
| R2_CORR | `test_state_r2.py` |
| R2_CPU (elementwise SIMD) | `test_elementwise_r2.py` (18) + `test_operations{,_extensive}.py` |
| R2_COEFFS | covered by `test_field.py` |

Bugs discovered during the R2E work are logged in `tests/BUG_LOG.md` per the
Testing & Bug Tracking protocol.

## Delivered audit items

- **Continuum-limit benchmark**, `benchmarks/r2_continuum_limit.py` (discrete `iΔ`
  vs continuum `iΔ` convention pin + the SJ Wightman positive-spectrum growth vs `n`).
- **Conference feature menu**, `documentation/guides/R2 Feature Menu.md`.
- **CI correctness gate**, `.github/workflows/ci.yml` runs `pytest tests/python`
  on the 3-OS matrix (Linux/macOS/Windows).

## Remaining (not yet audited/closed)

- **CI parity threshold** (R2_PERF: `benchmarks/r2_parity.py` with the `>= 0.90×`
  bar enforced in CI, timing thresholds are intentionally not CI-gated yet).
- **C++ R2E**, in progress: GPU `lu`/`qr`/`svd` factorizations are done (cuSOLVER,
  square dense, with CPU fallback); remaining are the shared `CudaLinalg` dispatch
  layer, CUDA wheel packaging, out-of-core `inverse`/`qr`/`svd` (tiled
  factorization driver), and R2_HARDEN polish (`__init__.py` slimming, CMake
  warning audit, macOS wheel portability).

## Deferred to future releases (explicitly NOT R2 scope)

Per the R2 field-theory boundary (Pauli–Jordan + retarded propagator is enough):

- Massive/Bessel Green's functions and the continuum Wightman log (R2_CONV/R2_CMVP
  ship the massless 1+1 pin only).
- Entanglement-entropy analytic reference (R2_ENT ships two documented conventions).
- RN/Kerr/Kerr–Newman black holes (R2_BH ships `Schwarzschild` 1+1).
- Higher-point Wick contractions, interacting fields, fermions, gauge fields
  (R2_FIELD/R2_CORR ship the free scalar core).
