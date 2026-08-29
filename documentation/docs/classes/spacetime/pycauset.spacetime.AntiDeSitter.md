# pycauset.spacetime.AntiDeSitter

```python
class AntiDeSitter(dimension=2, radius=1.0, rho_max=1.0)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

anti-de Sitter spacetime — the hyperboloid `−X₀² − X₁² + Σ Xᵢ² = −R²`.

## Parameters

*   **dimension** (*int*): Spacetime dimension `d` (≥ 2).
*   **radius** (*float*): The curvature radius `R`.
*   **rho_max** (*float*): The radial-coordinate cutoff `ρ ∈ [0, ρ_max]`.

## Notes (honest caveats)

- **No causal order:** the naive AdS hyperboloid has closed timelike curves, so `is_causal` raises `NotImplementedError` ("no causal order"); the universal cover is a research task.
- The **sampler** is a documented parametrization.
- `scalar_coeffs` raises `NotImplementedError` — coefficients are manual `(a, b)`.

## Methods

### is_causal

```python
def is_causal(self, u, v) -> bool
```

Raises `NotImplementedError` (naive AdS has no causal order).

### sample

```python
def sample(self, rng, n) -> np.ndarray
```

Draws `(n, d)` parametrized points `(t, ρ, …)`.

## See also

- [[docs/classes/spacetime/pycauset.spacetime.DeSitter.md|DeSitter]]
- [[docs/classes/spacetime/pycauset.spacetime.FLRW.md|FLRW]]
- [[guides/Spacetime.md|Spacetime guide]]
