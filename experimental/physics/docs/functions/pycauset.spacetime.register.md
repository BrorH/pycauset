# pycauset.spacetime.register

```python
@spacetime.register(name, *, overwrite=False)
```

Decorator that registers a `Spacetime` subclass under `name` (its persistable identity).

## Parameters

*   **name** (*str*): The registry key.
*   **overwrite** (*bool*): If `True`, replace an existing entry. Duplicate names raise `ValueError` by default (no silent last-wins).

## Example

```python
from pycauset import spacetime

@spacetime.register("my_diamond")
class MyDiamond(spacetime.Spacetime):
    ...
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]
- [[docs/functions/pycauset.spacetime.get_registry.md|spacetime.get_registry]]
- [[guides/Spacetime.md|Spacetime guide]]
