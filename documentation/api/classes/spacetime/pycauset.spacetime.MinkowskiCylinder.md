# pycauset.spacetime.MinkowskiCylinder

```python
class MinkowskiCylinder(dimension: int, height: float, circumference: float)
```

Inherits from: [[pycauset.spacetime.CausalSpacetime]]

Represents a flat Minkowski spacetime with periodic spatial boundary conditions ($S^1 \times \mathbb{R}$).

## Description

This topology is often used to study effects of spatial compactness. The spatial dimension wraps around, creating a cylinder-like structure in spacetime.

## Parameters

*   **dimension** (*int*): The dimension of the spacetime. Currently, only $d=2$ is supported.
*   **height** (*float*): The temporal duration of the region ($t \in [0, h]$).
*   **circumference** (*float*): The spatial circumference of the cylinder ($x \in [0, c]$).

## Properties

### height

```python
@property
def height(self) -> float
```

The temporal height of the cylinder.

### circumference

```python
@property
def circumference(self) -> float
```

The spatial circumference of the cylinder.

## Methods

### dimension

```python
def dimension(self) -> int
```

Returns the dimension of the spacetime.

### transform_coordinates

```python
def transform_coordinates(self, coords: np.ndarray) -> np.ndarray
```

*(Extension)* Transforms raw coordinates to a visualization-friendly basis.
For 2D, this maps $(t, x)$ to 3D cylindrical coordinates $(z, x, y)$.

### get_boundary

```python
def get_boundary(self) -> List[np.ndarray]
```

*(Extension)* Returns the boundary rings (top and bottom) of the cylinder in the transformed coordinate system.


### volume

```python
def volume(self) -> float
```

Returns the volume of the cylinder, calculated as $height \times circumference$.
