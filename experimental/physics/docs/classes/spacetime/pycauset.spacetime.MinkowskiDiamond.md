# pycauset.spacetime.MinkowskiDiamond

```python
class MinkowskiDiamond(dimension: int)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

Represents a causal diamond (Alexandrov interval) in flat Minkowski space.

## Description

A causal diamond is the intersection of the future of a point $p$ and the past of a point $q$, where $p \prec q$. In PyCauset, this is typically the standard unit diamond defined by the interval $[0, 1]^d$ in lightcone coordinates.

## Parameters

*   **dimension** (*int*): The dimension of the spacetime. Currently, only $d=2$ is fully supported for causal checks.

## Methods

### dimension

```python
def dimension(self) -> int
```

Returns the dimension of the spacetime.

### volume

```python
def volume(self) -> float
```

Returns the volume of the diamond. For the standard unit diamond in lightcone coordinates, the volume is normalized to 1.0.

## Spacetime contract methods

`MinkowskiDiamond` also implements the full `Spacetime` contract:

*   **signature** (*property*): `(1, d-1)`, Lorentzian.
*   **sample** (*method*): `sample(rng, n) -> np.ndarray`, draws `n` points uniformly in the diamond.
*   **is_causal** (*method*): `is_causal(u, v) -> bool`, the strict transitive causal order.
*   **is_causal_batch** (*method*): `is_causal_batch(coords) -> np.ndarray`, the vectorized `(n, n)` causal matrix.
*   **scalar_coeffs** (*method*): `scalar_coeffs(mass, density) -> (a, b)`, the authored 2D/4D field coefficients (raises `NotImplementedError` outside 2D/4D).
*   **to_embedding** / **boundary** (*methods*): presentation hooks.

See [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]] for the full contract.

