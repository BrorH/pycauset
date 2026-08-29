# pycauset.CausalSet

```python
class CausalSet(n=None, density=None, spacetime=None, seed=None, matrix=None, validate=True)
```

The `CausalSet` class represents a causal set — a discrete partial order. It is the primary object of PyCauset: a `TriangularBitMatrix` (the causal relations) plus provenance (the spacetime + seed it was sprinkled from, or an attached embedding).

## Parameters

*   **n** (*int, optional*): The number of elements to sprinkle. Either `n` or `density` is required.
*   **density** (*float, optional*): The sprinkling density; `n` is drawn as `Poisson(density × volume)`.
*   **spacetime** (*Spacetime, optional*): The spacetime to sprinkle into. Defaults to a 2D Minkowski diamond.
*   **seed** (*int | str, optional*): The random seed. Defaults to a random seed.
*   **matrix** (*TriangularBitMatrix, optional*): A pre-existing causal matrix (used by `load()`). If provided, sprinkling is skipped.
*   **validate** (*bool*): When `True` (default), a supplied `matrix` is validated as a strict partial order.

## Properties

### causal_matrix / C

```python
@property
def causal_matrix(self) -> TriangularBitMatrix
```

The causal matrix (`C[i, j] == 1` iff `i ≺ j`). `C` is an alias.

### n / N

```python
@property
def n(self) -> int
```

The number of elements.

### density / rho

```python
@property
def density(self) -> float
```

The sprinkling density, `n / volume`.

### spacetime

```python
@property
def spacetime(self) -> Spacetime
```

The spacetime used for sprinkling.

### embedding

```python
@property
def embedding(self) -> np.ndarray | None
```

The attached coordinate embedding (time-labelled), or `None` for a native (regenerated) causet.

## Methods

### validate

```python
def validate(self) -> None
```

Validates that the order is a strict partial order (reflexive-free, antisymmetric, transitive). Raises `ValueError` on the first violation.

### coordinates

```python
def coordinates(self, indices=None, force=False) -> np.ndarray
```

Returns the spacetime coordinates of the elements — the attached embedding for a custom-`Spacetime` causet, or regenerated from `(spacetime, seed)` for a native causet. Raises `UserWarning` above 100,000 elements unless `force=True`.

### links

```python
def links(self) -> np.ndarray
```

The link (Hasse) matrix — the transitive reduction `L = C & ~(C@C)`.

### past / future

```python
def past(self, x) -> np.ndarray
def future(self, x) -> np.ndarray
```

Indices `i` with `i ≺ x` (`past`) or `x ≺ i` (`future`).

### interval

```python
def interval(self, x, y) -> np.ndarray
```

The Alexandrov interval `I(x, y) = future(x) ∩ past(y)`.

### is_chain / is_antichain

```python
def is_chain(self, elements) -> bool
def is_antichain(self, elements) -> bool
```

Whether the given elements are pairwise comparable / pairwise incomparable.

### longest_chain

```python
def longest_chain(self) -> np.ndarray
```

A longest causal chain (indices); its length is the poset's height.

### layers

```python
def layers(self) -> list[np.ndarray]
```

Ranked layering: `layers[k]` holds the elements whose longest past-chain has length `k+1`.

### relation_fraction

```python
def relation_fraction(self) -> float
```

The fraction of element pairs that are causally related (`R / C(n, 2)`).

### myrheim_meyer_dimension

```python
def myrheim_meyer_dimension(self) -> float
```

The Myrheim–Meyer dimension estimate, inverting the relation fraction `f(d) = Γ(d+1)Γ(d/2)/(2Γ(3d/2))`.

### plot_embedding / plot_hasse / plot_causal_matrix

```python
def plot_embedding(self, **kwargs) -> Figure
def plot_hasse(self, **kwargs) -> Figure
def plot_causal_matrix(self, **kwargs) -> Figure
```

Visualization methods (lazy Plotly import). See [[docs/pycauset.vis/plot_embedding.md|plot_embedding]] et al.

### save / load

```python
def save(self, path) -> None
@staticmethod
def load(path) -> CausalSet
```

Save/load to/from the single-file `.pycauset` container.

## Example

```python
import pycauset as pc

c = pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=42)
c.validate()
links = c.links()
d = c.myrheim_meyer_dimension()
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]
- [[docs/classes/spacetime/pycauset.spacetime.MinkowskiDiamond.md|MinkowskiDiamond]]
- [[guides/Causal Sets.md|Causal Sets guide]]
- [[docs/functions/pycauset.causet.md|pc.causet]]
