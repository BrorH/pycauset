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
