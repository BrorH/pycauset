# pycauset.spacetime.TransformedSpacetime

```python
class TransformedSpacetime(base: Spacetime, forward: Callable, inverse: Callable | None = None)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

Wraps a spacetime with a coordinate transform. `forward` maps base coordinates to the wrapped chart (applied to `sample`); `inverse` maps back (applied to `is_causal`).

## Parameters

*   **base** (*Spacetime*): The spacetime to transform.
*   **forward** (*callable*): `coords -> coords` mapping base coordinates to the wrapped chart.
*   **inverse** (*callable, optional*): `coords -> coords` mapping back. Defaults to no-op (the transform then only affects the sampled coordinates, not the causal structure).

## Notes

The transform is assumed **volume-preserving** (e.g. a translation or rotation), so `volume()` returns `base.volume()`. For a non-volume-preserving map, subclass and override `volume()`.

## Example

```python
import numpy as np
from pycauset import spacetime

box = spacetime.MinkowskiBox(2, 10.0, 10.0)
shifted = spacetime.TransformedSpacetime(
    box,
    forward=lambda c: c + np.array([0.0, 3.0]),
    inverse=lambda c: c - np.array([0.0, 3.0]),
)
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.RestrictedSpacetime.md|RestrictedSpacetime]]
- [[project/plans/R2_SPACETIME_CREATION.md|R2 Spacetime Creation spec]]
