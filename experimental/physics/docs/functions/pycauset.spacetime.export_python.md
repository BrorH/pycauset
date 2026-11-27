# pycauset.spacetime.export_python

```python
spacetime.export_python(recipe_or_st) -> str
```

Emits a paste-ready `Spacetime` subclass for a recipe (or a `Spacetime` instance). The emitted class delegates to `create(recipe)`, the same template `create` uses, so it can never drift from the declarative builder.

## Parameters

*   **recipe_or_st** (*dict | Spacetime*): A recipe dict (with `name`, `dimension`, `domain`, `metric`, and domain params) or a built-in Minkowski spacetime.

## Returns

*   **str**: The generated Python source.

## Example

```python
from pycauset import spacetime

code = spacetime.export_python({"name": "MyBox", "dimension": 2, "domain": "box",
                                "time_extent": 2.0, "space_extent": 1.0})
ns = {"spacetime": spacetime}
exec(code, ns)
MyBox = ns["MyBox"]
```

## See also

- [[docs/functions/pycauset.spacetime.create.md|spacetime.create]]
- [[project/plans/R2_SPACETIME_CREATION.md|R2 Spacetime Creation spec]]
