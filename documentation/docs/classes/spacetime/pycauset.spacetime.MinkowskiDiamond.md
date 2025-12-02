# pycauset.spacetime.MinkowskiDiamond

```python
class MinkowskiDiamond(dimension: int)
```

Inherits from: [[pycauset.spacetime.CausalSpacetime]]

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

### transform_coordinates

```python
def transform_coordinates(self, coords: np.ndarray) -> np.ndarray
```

*(Extension)* Transforms raw lightcone coordinates to a visualization-friendly basis.
For 2D, this rotates $(u, v)$ to Cartesian $(t, x)$.

### get_boundary

```python
def get_boundary(self) -> List[np.ndarray]
```

*(Extension)* Returns the boundary of the spacetime region in the transformed coordinate system.
Used by visualization functions to draw the diamond edges.

