# pycauset.CausalSet

```py
class CausalSet(n=None, density=None, spacetime=None, seed=None, matrix=None, validate=True)
```

The `CausalSet` class represents a causal set, a discrete partial order. It is the primary object of PyCauset: a `TriangularMatrix` of dtype `bool` (the causal relations, one bit each) plus provenance (the spacetime and seed it was sprinkled from). Elements are labelled by time, so the matrix is strictly upper triangular.

## Parameters

*   **n** (*int, optional*): The number of elements to sprinkle. Either `n` or `density` is required.
*   **density** (*float, optional*): The sprinkling density; `n` is drawn as `Poisson(density × volume)`.
*   **spacetime** (*[[docs/classes/spacetime/pycauset.spacetime.MinkowskiDiamond.md|MinkowskiDiamond]], optional*): the region to sprinkle into. Defaults to the 2D diamond.
*   **seed** (*int | str, optional*): The random seed. Defaults to a random seed.
*   **matrix** (*TriangularMatrix (bool), optional*): A pre-existing causal matrix (used by `load()`). If provided, sprinkling is skipped.
*   **validate** (*bool*): When `True` (default), a supplied `matrix` is validated as a strict partial order.

## Properties

### causal_matrix / C

```py
@property
def causal_matrix(self) -> TriangularMatrix (bool)
```

The causal matrix (`C[i, j] == 1` iff `i ≺ j`). `C` is an alias.

### n / N

```py
@property
def n(self) -> int
```

The number of elements.

### density / rho

```py
@property
def density(self) -> float
```

The sprinkling density, `n / volume`.

### spacetime

```py
@property
def spacetime(self) -> MinkowskiDiamond
```

The spacetime used for sprinkling.

## Methods

### validate

```py
def validate(self) -> None
```

Validates that the order is a strict partial order (reflexive-free, antisymmetric, transitive). Raises `ValueError` on the first violation.

### coordinates

```py
def coordinates(self, indices=None, force=False) -> np.ndarray
```

Returns the spacetime coordinates of the elements, `(n, d)`, regenerated from `(spacetime, seed)` (the sprinkle's RNG stream) and time-labelled like the matrix. Raises `UserWarning` above 100,000 elements unless `force=True`.

### links

```py
def links(self) -> np.ndarray
```

The link (Hasse) matrix, the transitive reduction `L = C & ~(C@C)`.

### past / future

```py
def past(self, x) -> np.ndarray
def future(self, x) -> np.ndarray
```

Indices `i` with `i ≺ x` (`past`) or `x ≺ i` (`future`).

### interval

```py
def interval(self, x, y) -> np.ndarray
```

The Alexandrov interval `I(x, y) = future(x) ∩ past(y)`.

### is_chain / is_antichain

```py
def is_chain(self, elements) -> bool
def is_antichain(self, elements) -> bool
```

Whether the given elements are pairwise comparable / pairwise incomparable.

### longest_chain

```py
def longest_chain(self) -> np.ndarray
```

A longest causal chain (indices); its length is the poset's height.

### layers

```py
def layers(self) -> list[np.ndarray]
```

Ranked layering: `layers[k]` holds the elements whose longest past-chain has length `k+1`.

### relation_fraction

```py
def relation_fraction(self) -> float
```

The fraction of element pairs that are causally related (`R / C(n, 2)`).

### myrheim_meyer_dimension

```py
def myrheim_meyer_dimension(self) -> float
```

The Myrheim–Meyer dimension estimate, inverting the relation fraction `f(d) = Γ(d+1)Γ(d/2)/(2Γ(3d/2))`. Unbiased for a sprinkled Minkowski diamond in any dimension; the statistical error shrinks like `1/sqrt(n)`.

### save / load

```py
def save(self, path) -> None
@staticmethod
def load(path) -> CausalSet
```

Save/load to/from the single-file `.pycauset` container.

## Example

```python
import pycauset as pc

c = pc.causet(n=500, spacetime=pc.spacetime.MinkowskiDiamond(2), seed=42)
c.validate()
links = c.links()
d = c.myrheim_meyer_dimension()
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.MinkowskiDiamond.md|MinkowskiDiamond]]
- [[guides/Causal Sets.md|Causal Sets guide]]
- [[docs/functions/pycauset.causet.md|pc.causet]]
