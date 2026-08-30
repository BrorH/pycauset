# Tutorials

End-to-end walkthroughs. Each one is a complete script you can run as written. They
build on each other, so start at the top if you are new to the library.

## 1. Your first causal set

Sprinkle points into a spacetime, look at the order, and save the result.

```python
import numpy as np
import pycauset as pc

# 500 points in the default 2D Minkowski diamond.
c = pc.causet(n=500, seed=42)

print(c.n)                 # 500
print(c.coordinates().shape)   # (500, 2): time and space per element
print(c.C)                 # the causal matrix, a TriangularBitMatrix

# The causal matrix as a dense boolean array: C[i, j] is True when i is in j's past.
dense = np.asarray(c.C)
print(dense.dtype, dense.shape)

c.save("first.pycauset")
again = pc.load("first.pycauset")
print(again.n)
```

What happened: the points were labelled by time, so the causal matrix is strictly
upper triangular. `c.C` is bit-packed and disk-backed; `np.asarray(c.C)` densifies it
when you actually want the numbers.

## 2. Reading the causal structure

A causal set is a partial order. The structure methods live on `CausalSet` directly.

```python
import pycauset as pc

c = pc.causet(n=200, seed=7)

c.validate()                    # reflexive-free, antisymmetric, transitive

links = c.links()               # (n, n) boolean matrix of links (transitive reduction)
chain = c.longest_chain()       # indices of one longest chain
layers = c.layers()             # ranked layering of the poset

print(c.is_chain(list(chain)))  # True
print(chain)
print(len(layers))
```

`links()` is the Hasse skeleton: only immediate causal neighbours, not every related
pair. `longest_chain()` returns element indices, and `is_chain`/`is_antichain` check
any list of indices you pass.

## 3. Estimating the dimension of the underlying spacetime

Causal sets built from a $d$-dimensional Minkowski diamond have a relation fraction
that depends on $d$. The Myrheim-Meyer estimator inverts that.

```python
import pycauset as pc

for d in (2, 3):
    st = pc.spacetime.MinkowskiDiamond(dimension=d)
    c = pc.causet(n=300, spacetime=st, seed=42)
    print(f"built in {d}D -> relation fraction {c.relation_fraction():.3f}, "
          f"estimate {c.myrheim_meyer_dimension():.2f}")
```

The estimate comes back near the dimension you sprinkled into (about 1.96 for 2D,
2.88 for 3D at n=300). This is the discrete version of "measuring the dimension of
spacetime", and it is the kind of quantity causal set research extracts from the order
alone.

## 4. A scalar field and its propagators

Put a free scalar field on a causet and read off the propagators.

```python
import pycauset as pc

c = pc.causet(n=300, seed=11)
phi = pc.field("scalar", mass=1.0)
Q = phi.on(c)

K = Q.retarded()        # retarded Green's function K_R
iD = Q.pauli_jordan()   # iΔ = K_R - K_A
W = Q.wightman()        # Sorkin-Johnston vacuum two-point function

print(K.shape, iD.shape, W.shape)

S = Q.entanglement_entropy([0, 1, 2, 3, 4])
print(S)
```

`phi` is set-independent; `Q` is the field correlated on this particular causet. The
propagators come back as NumPy arrays. `entanglement_entropy(region)` takes a list of
element indices and returns a float. See [[guides/Field Theory|Field Theory]] for what
each function means and the coefficient conventions.

## 5. Synthetic orders, no geometry needed

Some questions are about the order itself, not a spacetime. `pycauset.synthetic`
builds them directly.

```python
import pycauset.synthetic as syn

chain = syn.chain(50)                          # a total order
anti = syn.antichain(50)                       # no relations at all
tp = syn.transitive_percolation(0.2, 100)      # random order, p first, then n
grid = syn.product_order((8, 8))               # a 2D grid order
```

```python
print(chain.is_chain(list(range(50))))         # True
print(anti.is_antichain(list(range(50))))      # True
print(tp.n, grid.n)
```

Note the argument order on `transitive_percolation` and `random_dag_order`: the
probability `p` comes first, then `n`.

## 6. Linear algebra without the physics

The matrix engine underneath the physics is usable on its own.

```python
import pycauset as pc

A = pc.matrix([ [4.0, 1.0], [1.0, 3.0] ], dtype="float64")
b = pc.matrix([ [1.0], [2.0] ], dtype="float64")

x = pc.solve(A, b)
vals, vecs = pc.eig(A)
U, s, Vh = pc.svd(A)

print(pc.to_numpy(x).ravel())
print(pc.to_numpy(vals))
```

`pc.to_numpy(...)` materializes a result (or a lazy expression) as a NumPy array. See
[[guides/Linear Algebra Operations|Linear Algebra Operations]] for the full surface.

## Where next

- [[guides/User Guide|User Guide]] — the same ground, organized by concept.
- [[guides/Examples|Examples]] — short, copy-paste recipes.
- [[guides/Spacetime|Spacetime]] — the spacetime library and custom spacetimes.
- [[guides/Visualization|Visualization]] — embedding, Hasse, and heatmap plots.
