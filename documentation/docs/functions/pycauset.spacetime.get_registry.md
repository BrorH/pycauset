# pycauset.spacetime.get_registry

```python
spacetime.get_registry() -> dict[str, type]
```

Returns a shallow copy of the spacetime name registry (name → `Spacetime` subclass).

## Example

```python
from pycauset import spacetime

registry = spacetime.get_registry()
# includes the built-ins: "minkowski_diamond", "minkowski_cylinder", "minkowski_box"
```

## See also

- [[docs/functions/pycauset.spacetime.register.md|spacetime.register]]
- [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]
