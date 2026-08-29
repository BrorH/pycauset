# pycauset.spacetime.Spacetime

```python
class Spacetime
```

`Spacetime` is the abstract base class that makes custom spacetimes first-class. Subclass it and implement four methods — `dimension()`, `volume()`, `sample(rng, n)`, and `is_causal(u, v)` — and you get a spacetime the sprinkler, field engine, and visualizer all understand.

## Description

A `Spacetime` bundles a continuum region, a measure, and a causal order:

- `signature` — the first-class signature `(t, s) = (timelike, spacelike)`. It defaults to Lorentzian `(1, d-1)`; declare a class attribute `signature = (t, s)` to override.
- `sample()` — draws points uniformly w.r.t. the measure whose mass is `volume()`. It must be a pure function of the injected RNG (this is what makes `same (spacetime, seed, n) ⇒ bit-identical` hold).
- `is_causal()` — the strict, transitive partial order (the closure, not the links). A causal order exists only for Lorentzian `t == 1`; the base implementation raises for any other signature rather than guessing.

## Signature property

```python
@property
def signature(self) -> tuple[int, int]
```

Returns `(t, s)`. Defaults to Lorentzian `(1, dimension() - 1)`. A subclass that declares `signature = (t, s)` as a class attribute shadows this property.

## Methods

### dimension

```python
@abstractmethod
def dimension(self) -> int
```

Total spacetime dimension `d = t + s` (index 0 is time for Lorentzian signatures).

### volume

```python
@abstractmethod
def volume(self) -> float
```

Total mass of the sampling measure (`0 < volume() < inf`).

### sample

```python
@abstractmethod
def sample(self, rng, n) -> np.ndarray
```

Draws `n` points as an `(n, d)` array, uniform w.r.t. the measure whose mass is `volume()`.

### is_causal

```python
def is_causal(self, u, v) -> bool
```

Strict partial order (the transitive closure), element-wise. The base implementation raises for any signature with `t != 1`: Euclidean `(0, d)` spacetimes are point processes, and multi-time `(t > 1, s)` spacetimes must supply their own "future" convention.

### is_causal_batch

```python
def is_causal_batch(self, coords) -> np.ndarray
```

Optional fast path: the `(n, n)` boolean causal matrix (upper-triangular). The sprinkler uses it when present, else falls back to element-wise `is_causal`.

### scalar_coeffs

```python
def scalar_coeffs(self, mass, density) -> tuple[float, float]
```

Authored field coefficients `(a, b)`, or raise — never guessed. The built-in Minkowski spacetimes implement the known 2D/4D table.

### to_embedding / boundary / display_axes

```python
def to_embedding(self, coords) -> np.ndarray
def boundary(self) -> list[np.ndarray]
def display_axes(self) -> list[str] | None
```

Presentation hooks (default: identity / none / none). The visualizer reads these
authored declarations and never guesses geometry. `display_axes` returns one axis
label per embedding column, or `None` for the generic `c0, c1, …` fallback.

## Example

```python
from pycauset import spacetime

@spacetime.register("my_diamond")
class MyDiamond(spacetime.Spacetime):
    def dimension(self): return 2
    def volume(self): return 1.0
    def sample(self, rng, n): return rng.uniform(0.0, 1.0, size=(n, 2))
    def is_causal(self, u, v): return u[0] < v[0] and u[1] < v[1]
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.MinkowskiDiamond.md|MinkowskiDiamond]]
- [[docs/functions/pycauset.spacetime.register.md|spacetime.register]]
- [[guides/Spacetime.md|Spacetime guide]]
- [[project/plans/R2_SPACETIME_CREATION.md|R2 Spacetime Creation spec]]
