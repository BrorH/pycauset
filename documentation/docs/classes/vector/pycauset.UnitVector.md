# pycauset.UnitVector

```python
class pycauset.UnitVector(n, active_index)
```

The standard basis vector `e_k` of length `n` (1.0 at `active_index`, 0.0 elsewhere).
Stores no elements and is read-only; dtype `float64`.

## Example

```python
import pycauset as pc

e = pc.UnitVector(4, 2)
assert e[2] == 1.0 and e[0] == 0.0 and e.active_index == 2
```
