# pycauset.field.ContinuumCorrelatedField

```python
class ContinuumCorrelatedField(spacetime: Spacetime, mass: float = 0.0)
```

Closed-form Green's functions on a continuum flat-Minkowski spacetime, for the R2 continuum-limit comparison. Returned by `phi.on(spacetime)`.

## Description

Provides the continuum `G_R`, `G_A`, and `iΔ` as callables of two points, plus `.at(coords)` sampling. Currently exact for the **massless 1+1** case (`iΔ = (i/2) sgn(Δt) θ(σ)`); massive and higher-dimensional closed forms (Bessel functions) are pending.

## Methods

### retarded / advanced

```python
def retarded(self, x, y) -> float
def advanced(self, x, y) -> float
```

Continuum retarded/advanced Green's function at two points.

### pauli_jordan

```python
def pauli_jordan(self, x, y) -> complex
```

Continuum Pauli–Jordan `iΔ(x, y) = i(G_R - G_A)`.

### at

```python
def at(self, coords, which="pauli_jordan") -> np.ndarray
```

Samples a kernel at an `(n, d)` coordinate array, returning an `(n, n)` matrix.

## Example

```python
import pycauset as pc

c = pc.causet(n=100, spacetime=pc.MinkowskiDiamond(2), seed=1)
Q_ct = pc.field("scalar", mass=0.0).on(c.spacetime)
iD_continuum = Q_ct.at(c.spacetime.to_embedding(c.embedding), which="pauli_jordan")
```

## See also

- [[docs/classes/field/pycauset.field.CorrelatedField.md|CorrelatedField]]
- [[guides/Field Theory.md|Field Theory guide]]
