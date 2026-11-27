# pycauset.field.CorrelatedField

```python
class CorrelatedField(causet: CausalSet, mass: float = 0.0)
```

A field together with its Green's functions and vacuum two-point function on a causal set. Returned by `phi.on(causet)` for a `Field`.

## Parameters

*   **causet** (*CausalSet*): The background causal set.
*   **mass** (*float*): The field mass. Defaults to 0.0 (massless).

## Methods

### retarded

```python
def retarded(self, a=None, b=None) -> np.ndarray
```

Retarded Green's function `K_R = aC (I - baC)⁻¹` as a dense `n×n` array. `a`, `b` default to `Spacetime.scalar_coeffs(mass, density)`; pass them manually to override.

### advanced

```python
def advanced(self, a=None, b=None) -> np.ndarray
```

Advanced Green's function `K_A = K_Rᵀ`.

### pauli_jordan

```python
def pauli_jordan(self) -> np.ndarray
```

Pauli–Jordan function `iΔ = K_R - K_A` (a Hermitian matrix).

### wightman

```python
def wightman(self) -> np.ndarray
```

Sorkin–Johnston vacuum Wightman two-point function, the positive-eigenvalue part of `iΔ`.

### correlator

```python
def correlator(self) -> np.ndarray
```

Vacuum two-point function `⟨φφ⟩ = W` (free field).

### state

```python
def state(self, config=None) -> State
```

Builds a [[docs/classes/field/pycauset.field.State.md|State]] (a coherent field configuration) on top of this correlated field. `config` defaults to the vacuum (zero).

### entanglement_entropy

```python
def entanglement_entropy(self, region, convention="sorkin_yazdi") -> float
```

Sorkin–Yazdi entanglement entropy of a region (a subset of element indices), from the region-restricted SJ Wightman matrix.

**Conventions (documented):**

| `convention` | formula | requirement |
| :-- | :-- | :-- |
| `"sorkin_yazdi"` *(default)* | `S = tr[(W_A + I) ln(W_A + I) − W_A ln W_A]` (`0 ln 0 = 0`) | `W_A ≥ 0`, the SJ Wightman's zero-point "1/2" convention |
| `"symplectic"` | `S = tr[(W_A + 1/2) ln(W_A + 1/2) − (W_A − 1/2) ln(W_A − 1/2)]` | `W_A ≥ 1/2` (raises `ValueError` otherwise) |

The two conventions are the same formula up to the zero-point shift:
`sorkin_yazdi(W) == symplectic(W + 1/2 I)`.

## Example

```python
import pycauset as pc

c = pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=42)
Q = pc.field("scalar", mass=1.0).on(c)
W = Q.wightman()
```

## See also

- [[docs/classes/field/pycauset.field.Field.md|Field]]
- [[docs/classes/field/pycauset.field.State.md|State]]
- [[guides/Field Theory.md|Field Theory guide]]
