# User Guide

A guided tour of PyCauset. It walks through one causal set from creation to a field
on top of it, and points at the pieces as they come up. It is the page to read once
so the rest of the guides make sense.

## The mental model

PyCauset has three layers:

1. **Geometry.** A `CausalSet` is a finite set of elements with a causal order
   between them. You make one by *sprinkling* points into a `Spacetime` and reading
   the order off the geometry.
2. **Matter.** A `Field` is set-independent; apply it to a causal set and you get a
   `CorrelatedField` with propagators on that set.
3. **The engine.** Underneath both is a matrix/vector system that stores data in RAM
   or on disk and does the linear algebra. You can use it directly without touching
   the physics.

## Make a causal set

```python
import pycauset as pc

c = pc.causet(n=1000, seed=42)
```

That sprinkles 1000 points into the default region, a 2D Minkowski diamond, and
stores the causal order between them. Points are labelled by time, so the order is a
strictly upper-triangular matrix.

Choose the region with the `spacetime` argument:

```python
st = pc.spacetime.MinkowskiDiamond(dimension=4)
c = pc.causet(n=5000, spacetime=st, seed=1)
```

Sprinkle a *density* instead of a fixed count, and the number of points becomes a
Poisson draw:

```python
c = pc.causet(density=500, seed=1)   # N ~ Poisson(rho * volume)
```

## Look inside it

```python
c.n                   # number of elements
c.coordinates()       # (n, 2) array of points
c.C                   # the causal matrix, a TriangularBitMatrix
```

`c.C[i, j]` is `True` when element `i` is in the causal past of `j`. It is bit-packed
and disk-backed; get the dense boolean array when you actually want it:

```python
import numpy as np

dense = np.asarray(c.C)
```

View the same matrix as a heatmap:

```python
c.plot_causal_matrix().show()
```

A bright cell means `i` is in the past of `j`; the upper-triangular pattern is the
signature of a time-labelled causet. See [[guides/Visualization|Visualization]] for
this and the other plots.

`c.validate()` checks the order is reflexive-free, antisymmetric, and transitive. The
constructor already did this, so it should pass.

## Analyse the order

A causal set is a partial order, and its structure methods are first-class:

```python
links = c.links()              # (n, n) boolean matrix of links (transitive reduction)
chain = c.longest_chain()      # indices of one longest chain
layers = c.layers()            # ranked layering

c.is_chain(list(chain))        # True
c.is_antichain([0, 1, 2])      # depends on the set
```

`links()` is the Hasse skeleton: only immediate causal neighbours, not every related
pair.

Two quantities a physicist reaches for straight away:

```python
c.relation_fraction()          # |relations| / max possible, in [0, 1]
c.myrheim_meyer_dimension()    # dimension estimate from the order alone
```

The relation fraction of a causet sprinkled into a $d$-dimensional diamond is
characteristic of $d$; Myrheim-Meyer inverts that. See
[[guides/Tutorials|Tutorials]] for the walkthrough.

## Plot it

```python
c.plot_embedding().show()      # points in spacetime
c.plot_hasse().show()          # links only
c.plot_causal_matrix().show()  # the order as a heatmap
pc.show(c)                     # embedding plot, then .show(), in one call
```

See [[guides/Visualization|Visualization]] for what each looks like.

## Put a field on it

```python
phi = pc.field("scalar", mass=1.0)   # a Field, set-independent
Q = phi.on(c)                        # a CorrelatedField on this causet

K = Q.retarded()          # retarded propagator K_R
iD = Q.pauli_jordan()     # iΔ = K_R - K_A
W = Q.wightman()          # Sorkin-Johnston vacuum two-point function
S = Q.entanglement_entropy([0, 1, 2, 3, 4])
```

The propagators come back as NumPy arrays. See
[[guides/Field Theory|Field Theory]] for what each one means and how the coefficients
are set.

## Save and load

```python
c.save("universe.pycauset")
again = pc.load("universe.pycauset")
```

One file holds the causal matrix, coordinates, and metadata, so `pc.load` reconstructs
the object without re-sprinkling.

## Use the matrix engine directly

The same storage and dispatch run underneath. You can skip the physics:

```python
A = pc.matrix([ [4.0, 1.0], [1.0, 3.0] ], dtype="float64")
b = pc.matrix([ [1.0], [2.0] ], dtype="float64")

x = pc.solve(A, b)           # solve A x = b
pc.to_numpy(x)               # -> NumPy array

vals, vecs = pc.eigh(A)      # symmetric eigenvalues/vectors
```

Operations are lazy where it matters: `A + B` builds an expression, and the work runs
when you materialize it (element access, `pc.to_numpy(...)`, or `pc.save(...)`).

## Work with NumPy

```python
m = pc.matrix(np.array([ [1, 2], [3, 4] ]))   # from NumPy
back = np.asarray(m)                          # to NumPy
```

Small objects behave like NumPy; large ones spill to disk automatically. See
[[guides/Numpy Integration|NumPy Integration]] for the interop rules and the
materialization guard.

## Scale up

When the data outgrows RAM, PyCauset moves it to disk instead of failing. Two knobs
cover most of it:

```python
pc.set_backing_dir("./pycauset_storage")    # where disk-backed files go
pc.set_memory_threshold(100 * 1024 * 1024)  # spill above 100 MB
pc.set_num_threads(4)                       # cap parallel work
```

See [[guides/Storage and Memory|Storage and Memory]],
[[guides/Performance Guide|Performance]], and
[[guides/Advanced Usage|Advanced Usage]] for the full set of controls.

## Where next

- [[guides/Tutorials|Tutorials]] — specific walkthroughs, run end to end.
- [[guides/Examples|Examples]] — copy-paste recipes.
- [[docs/index|API Reference]] — exact signatures.
