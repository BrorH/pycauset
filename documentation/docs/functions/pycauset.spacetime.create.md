# pycauset.spacetime.create

```python
spacetime.create(name=None, *, dimension, signature=None, domain, metric="flat", **params) -> Spacetime
```

The Rung-0 declarative builder: assemble a configured `Spacetime` from a recipe, with no class. Every parameter maps 1:1 to a concrete setting, there is no hidden inference.

## Parameters

*   **name** (*str, optional*): A registry name (for persistence).
*   **dimension** (*int*): Total spacetime dimension (≥ 2).
*   **signature** (*tuple, optional*): `(t, s)`. Defaults to Lorentzian `(1, d-1)`.
*   **domain** (*str*): `"diamond"`, `"cylinder"`, or `"box"`.
*   **metric** (*str*): `"flat"` (only; curved metrics raise `NotImplementedError`).
*   **params**: Domain-specific, `height`/`circumference` for `"cylinder"`, `time_extent`/`space_extent` for `"box"`.

## Raises

*   **ValueError**: invalid `dimension`/`signature`, or missing domain parameters.
*   **NotImplementedError**: unsupported `metric`/`domain`.

## Example

```python
from pycauset import spacetime

st = spacetime.create(dimension=3, domain="box", time_extent=2.0, space_extent=3.0)
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]
- [[docs/functions/pycauset.spacetime.export_python.md|spacetime.export_python]]
- [[guides/Spacetime.md|Spacetime guide]]
