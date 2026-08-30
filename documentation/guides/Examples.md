# Examples

Short, copy-paste recipes. Each block is self-contained. Run them with
`import pycauset as pc` and (where needed) `import numpy as np` at the top of your
script.

## Sprinkle a causal set

```python
import pycauset as pc

c = pc.causet(n=1000, seed=42)        # 1000 points, default 2D Minkowski diamond
c = pc.causet(density=500, seed=1)    # Poisson: N ~ Poisson(rho * V), volume 1
c = pc.causet(n=1000, seed="run_1")   # string seeds work too
```

## Pick a spacetime

```python
pc.causet(n=500, spacetime=pc.spacetime.MinkowskiDiamond(dimension=4))
pc.causet(n=500, spacetime=pc.spacetime.MinkowskiBox(2, time_extent=2.0, space_extent=1.0))
pc.causet(n=500, spacetime=pc.spacetime.MinkowskiCylinder(2, height=2.0, circumference=5.0))
pc.causet(n=500, spacetime=pc.spacetime.DeSitter(2))
pc.causet(n=500, spacetime=pc.spacetime.AntiDeSitter(2))
pc.causet(n=500, spacetime=pc.spacetime.FLRW(2))
pc.causet(n=500, spacetime=pc.spacetime.Schwarzschild(2))
```

`MinkowskiDiamond`, `MinkowskiBox`, and `MinkowskiCylinder` take a `dimension` and a
couple of extent parameters; the curved spacetimes ship as parametrizations with
manual field coefficients. See [[guides/Spacetime|Spacetime]].

## Read the causal matrix

```python
import numpy as np

C = c.C                       # TriangularBitMatrix (bit-packed, disk-backed)
dense = np.asarray(C)         # bool array, C[i, j] is True when i < j
```

## Causal structure

```python
c.validate()                        # strict partial order check (returns None)
links = c.links()                   # (n, n) boolean matrix of links
chain = c.longest_chain()           # indices of a longest chain
layers = c.layers()                 # ranked layering
past = c.past(x)                    # indices i with i < x
future = c.future(x)                # indices j with x < j
interval = c.interval(x, y)         # indices in future(x) ∩ past(y)
c.is_chain([0, 1, 2])               # bool
c.is_antichain([0, 1, 2])           # bool
c.myrheim_meyer_dimension()         # float, dimension estimate
c.relation_fraction()               # float, |relations| / max possible
```

## Put a scalar field on it

```python
phi = pc.field("scalar", mass=1.0)
Q = phi.on(c)

K = Q.retarded()                    # K_R
iD = Q.pauli_jordan()               # iΔ = K_R - K_A
W = Q.wightman()                    # Sorkin-Johnston vacuum
S = Q.entanglement_entropy([0, 1, 2, 3])
```

Massless: `pc.field("scalar", mass=0.0)`. The continuum limit is
`phi.on(spacetime)`, which returns a `ContinuumCorrelatedField` with a `.at(coords)`
sampler.

## Synthetic orders

```python
import pycauset.synthetic as syn

syn.chain(50)                          # total order
syn.antichain(50)                      # no relations
syn.transitive_percolation(0.2, 100)   # p first, then n
syn.random_dag_order(0.3, 100)         # p first, then n
syn.product_order((8, 8))              # grid order
syn.poset([(0, 1), (1, 2)], n=3)       # explicit relations
```

## Define a custom spacetime

```python
from pycauset import spacetime

@spacetime.register("my_diamond")
class MyDiamond(spacetime.Spacetime):
    def dimension(self): return 2
    def volume(self): return 1.0
    def sample(self, rng, n): return rng.uniform(0.0, 1.0, size=(n, 2))
    def is_causal(self, u, v): return u[0] < v[0] and u[1] < v[1]

c = pc.causet(n=500, spacetime=MyDiamond(), seed=42)
```

Composition decorators build new spacetimes from an existing one:

```python
box = pc.spacetime.MinkowskiBox(2, 2.0, 2.0)
half = pc.spacetime.RestrictedSpacetime(box, region=lambda x: x[1] < 1.0)
blown = pc.spacetime.ConformalSpacetime(box, conformal_factor=lambda x: 2.0)
ring = pc.spacetime.PeriodicSpacetime(box, periods={1: 1.0})
```

Declarative construction and codegen:

```python
st = pc.spacetime.create(dimension=2, domain="diamond")
code = pc.spacetime.export_python(st)
```

## Plot

```python
pc.plot_embedding(c).show()
c.plot_hasse().show()
c.plot_causal_matrix().show()
pc.show(c)                           # embedding plot + .show() in one call
```

## Save and load

```python
c.save("universe.pycauset")
again = pc.load("universe.pycauset")
```

## Linear algebra

```python
A = pc.matrix([ [4.0, 1.0], [1.0, 3.0] ], dtype="float64")
b = pc.matrix([ [1.0], [2.0] ], dtype="float64")

x = pc.solve(A, b)                   # solve A x = b
pc.to_numpy(x)                       # -> NumPy array

M = pc.matrix([ [1.0, 2.0], [3.0, 4.0] ], dtype="float64")
P, L, U = pc.lu(M)                   # LU decomposition, (P, L, U)

spd = pc.matrix([ [4.0, 1.0], [1.0, 3.0] ], dtype="float64")
Lc = pc.cholesky(spd)                # lower-triangular Cholesky factor
vals, vecs = pc.eigh(spd)            # symmetric eigenvalues/vectors
U, s, Vh = pc.svd(spd)               # singular values

pc.invert(spd)                       # inverse
pc.cond(spd)                         # condition number
pc.slogdet(spd)                      # (sign, logabsdet)
```

Eigenvalue and SVD results are disk-backed; convert them with
`pc.to_numpy(..., allow_huge=True)` or `np.asarray(...)` where supported. See
[[guides/Linear Algebra Operations|Linear Algebra Operations]] for the full surface
and the exact return types.

## NumPy interop

```python
m = pc.matrix(np.array([ [1, 2], [3, 4] ]))   # from NumPy
back = np.asarray(m)                        # to NumPy

v = pc.vector([1.0, 2.0, 3.0])              # vectors are 1xN or Nx1 matrices
```

## Configuration

```python
pc.set_backing_dir("./pycauset_storage")    # where disk-backed files go
pc.set_memory_threshold(100 * 1024 * 1024)  # spill above 100 MB
pc.set_num_threads(4)                       # cap parallel work
```

See [[guides/Advanced Usage|Advanced Usage]] for the full set of knobs.
