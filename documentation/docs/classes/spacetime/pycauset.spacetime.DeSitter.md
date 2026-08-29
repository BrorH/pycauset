# pycauset.spacetime.DeSitter

```python
class DeSitter(dimension=2, radius=1.0, time_extent=2.0)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

de Sitter spacetime — the hyperboloid `−X₀² + Σ Xᵢ² = R²` in (d+1)-dim Minkowski.

## Parameters

*   **dimension** (*int*): Spacetime dimension `d` (≥ 2).
*   **radius** (*float*): The curvature radius `R`.
*   **time_extent** (*float*): The global-time patch `t ∈ [-T, T]`.

## Notes (honest caveats)

- The **sampler** is a *documented parametrization* (uniform in global time `t` and spherical angles), **not** the dS-invariant measure.
- `is_causal` is the ambient-Minkowski causal order restricted to the hyperboloid.
- `scalar_coeffs` raises `NotImplementedError` — coefficients are manual `(a, b)`.

## Methods

### is_causal

```python
def is_causal(self, u, v) -> bool
```

`True` iff `v` is in the ambient future light cone of `u` (timelike/null).

### sample / volume

```python
def sample(self, rng, n) -> np.ndarray
def volume(self) -> float
```

`sample` draws `(n, d)` global-coordinate points; `volume` is the finite-patch volume `Rᵈ · A_{d-1} · ∫ cosh^{d-1}(t) dt`.

## Example

```python
import pycauset as pc
from pycauset import spacetime

c = pc.causet(n=500, spacetime=spacetime.DeSitter(2), seed=1)
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.AntiDeSitter.md|AntiDeSitter]]
- [[docs/classes/spacetime/pycauset.spacetime.FLRW.md|FLRW]]
- [[guides/Spacetime.md|Spacetime guide]]
