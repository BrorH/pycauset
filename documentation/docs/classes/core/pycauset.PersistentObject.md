# pycauset.PersistentObject

Base class for all persistent objects in pycauset (Matrices and Vectors).

## Properties

### `scalar`
Get the scalar value associated with this object, if any.

### `seed`
Get the random seed used to generate this object, if applicable.

### `is_temporary`
Check if the object is backed by a temporary file that will be deleted on close.

## Methods

### `close()`
Release the memory-mapped backing file. The object becomes unusable afterward.

### `get_backing_file()`
Get the path to the memory-mapped file backing this object.
