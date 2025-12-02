# pycauset.field.Field

```python
class Field(abc.ABC)
```

Abstract base class for all fields defined on a Causal Set.

A `Field` represents the matter content (or vacuum state) imposed on the spacetime geometry of a `CausalSet`. It separates the physical field parameters (like mass) from the geometric parameters of the set itself.

## Properties

### causet
```python
@property
causet: CausalSet
```
The causal set instance on which this field is defined.

## Methods

### propagator
```python
@abc.abstractmethod
def propagator(self) -> MatrixBase
```
Computes the propagator (Green's function) for this field. The specific type of propagator (Retarded, Feynman, etc.) depends on the subclass implementation.
