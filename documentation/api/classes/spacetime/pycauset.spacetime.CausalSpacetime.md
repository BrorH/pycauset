# pycauset.spacetime.CausalSpacetime

```python
class CausalSpacetime
```

The abstract base class for all spacetime manifolds in PyCauset.

## Description

This class defines the interface that all spacetime implementations must adhere to. It allows the `Sprinkler` to generate points and determine causal relations without knowing the details of the underlying geometry.

## Methods

### dimension

```python
def dimension(self) -> int
```

Returns the number of spacetime dimensions (e.g., 2 for 1+1 dimensions).

### volume

```python
def volume(self) -> float
```

Returns the total spacetime volume of the region represented by this object. This is used by the sprinkler to calculate the expected number of elements for a given density.
