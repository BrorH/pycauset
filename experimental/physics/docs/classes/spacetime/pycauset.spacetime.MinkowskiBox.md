# pycauset.spacetime.MinkowskiBox

```python
class MinkowskiBox(dimension: int, time_extent: float, space_extent: float)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

Represents a rectangular region (block) in flat Minkowski space with hard boundaries.

## Description

Unlike the `MinkowskiDiamond`, which is defined by null boundaries (light rays), the `MinkowskiBox` is defined by coordinate planes. This is useful for studying finite-size effects with spatial boundaries.

## Parameters

*   **dimension** (*int*): The dimension of the spacetime.
*   **time_extent** (*float*): The duration of the region in the time coordinate ($t \in [0, T]$).
*   **space_extent** (*float*): The length of the region in the spatial coordinates ($x_i \in [0, L]$).

## Properties

### time_extent

```python
@property
def time_extent(self) -> float
```

The temporal extent of the box.

### space_extent

```python
@property
def space_extent(self) -> float
```

The spatial extent of the box.

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

Returns the volume of the box ($T \times L^{d-1}$).

## Spacetime contract methods

`MinkowskiBox` also implements the full `Spacetime` contract:

*   **signature** (*property*): `(1, d-1)`, Lorentzian.
*   **sample** (*method*): `sample(rng, n) -> np.ndarray`, draws `n` points uniformly in the box.
*   **is_causal** (*method*): `is_causal(u, v) -> bool`, the strict transitive causal order (`dt > ||dx||`).
*   **is_causal_batch** (*method*): `is_causal_batch(coords) -> np.ndarray`, the vectorized `(n, n)` causal matrix.
*   **scalar_coeffs** (*method*): `scalar_coeffs(mass, density) -> (a, b)`, the authored 2D/4D field coefficients.
*   **to_embedding** / **boundary** (*methods*): presentation hooks.

See [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]] for the full contract.
