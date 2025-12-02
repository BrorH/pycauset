# pycauset.spacetime.MinkowskiBox

```python
class MinkowskiBox(dimension: int, time_extent: float, space_extent: float)
```

Inherits from: [[pycauset.spacetime.CausalSpacetime]]

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

### transform_coordinates

```python
def transform_coordinates(self, coords: np.ndarray) -> np.ndarray
```

*(Extension)* Returns the coordinates as-is (identity transform), as they are already Cartesian.

### get_boundary

```python
def get_boundary(self) -> List[np.ndarray]
```

*(Extension)* Returns the rectangular boundary of the box for visualization.
