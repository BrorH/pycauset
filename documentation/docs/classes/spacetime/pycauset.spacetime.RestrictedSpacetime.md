# pycauset.spacetime.RestrictedSpacetime

```python
class RestrictedSpacetime(base: Spacetime, region: Callable, volume: float | None = None)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

Wraps a spacetime and keeps a subregion defined by a predicate `region(coords) -> bool`. Sampling uses rejection; `is_causal` is inherited unchanged.

## Parameters

*   **base** (*Spacetime*): The spacetime to restrict.
*   **region** (*callable*): A predicate `region(coords) -> bool` selecting the kept points.
*   **volume** (*float, optional*): The restricted volume. If omitted, it is estimated by Monte Carlo so `volume ↔ sample` stay consistent.

## Methods

### volume

```python
def volume(self) -> float
```

The restricted volume (provided, or Monte-Carlo estimated).

### sample

```python
def sample(self, rng, n) -> np.ndarray
```

Draws `n` points uniformly from the subregion by rejection sampling the base.

### is_causal

```python
def is_causal(self, u, v) -> bool
```

Inherited from `base`.

## Example

```python
from pycauset import spacetime

box = spacetime.MinkowskiBox(2, 10.0, 10.0)
left_half = spacetime.RestrictedSpacetime(box, region=lambda c: c[1] < 5.0)
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.TransformedSpacetime.md|TransformedSpacetime]]
- [[project/plans/R2_SPACETIME_CREATION.md|R2 Spacetime Creation spec]]
