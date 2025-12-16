# pycauset.VectorBase

Base class for all vector types. Inherits from [PersistentObject](../core/pycauset.PersistentObject.md).

## Properties

### `T`
Returns the transpose of the vector. If the vector is a column vector (default), returns a row vector, and vice versa.

## Methods

### `__array__()`
Convert the vector to a NumPy array. This allows pycauset vectors to be passed directly to NumPy functions.
