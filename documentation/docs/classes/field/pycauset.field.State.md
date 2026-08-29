# pycauset.field.State

```python
class State(correlated_field: CorrelatedField, config: np.ndarray)
```

A specific excitation of the vacuum, a coherent/classical field configuration over the causet, carrying the vacuum two-point function. Returned by `CorrelatedField.state(config)`.

## Description

For a Gaussian/coherent state with configuration `phi`:

- `⟨φ⟩ = phi`
- `⟨φφ⟩ = phi phiᵀ + W`
- `⟨φ²⟩ = diag(W) + phi²`

where `W` is the Sorkin–Johnston Wightman function.

## Methods

### field

```python
def field(self) -> np.ndarray
```

`⟨φ⟩`, the mean field configuration.

### two_point

```python
def two_point(self) -> np.ndarray
```

`⟨φφ⟩ = phi phiᵀ + W`.

### field_variance

```python
def field_variance(self) -> np.ndarray
```

`⟨φ²⟩ = diag(W) + phi²` (per-element fluctuation).

## Example

```python
import numpy as np
import pycauset as pc

c = pc.causet(n=100, spacetime=pc.MinkowskiDiamond(2), seed=1)
Q = pc.field("scalar", mass=1.0).on(c)
state = Q.state(np.arange(c.n, dtype=float))
vev_phi_sq = state.field_variance()
```

## See also

- [[docs/classes/field/pycauset.field.CorrelatedField.md|CorrelatedField]]
- [[guides/Field Theory.md|Field Theory guide]]
