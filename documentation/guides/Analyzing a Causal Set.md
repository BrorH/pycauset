# Analyzing a Causal Set

The order is the data. Once you have a causal set, these are the methods for reading
what the order says. They all live on `CausalSet`.

## Check it is a valid order

```python
import pycauset as pc

c = pc.causet(n=500, seed=42)
c.validate()          # reflexive-free, antisymmetric, transitive; returns None
```

The constructor already validated it, so `validate()` is for re-checking after you
have done something to the set.

## The links (transitive reduction)

`links()` returns the Hasse skeleton: only immediate causal neighbours, not every
related pair.

```python
links = c.links()     # (n, n) boolean matrix
```

If `i < j` and there is no `k` with `i < k < j`, then `links[i, j]` is `True`. The
causal matrix `c.C` has every related pair; `links` has only the ones with nothing in
between.

## Chains, antichains, layers

```python
chain = c.longest_chain()   # indices of one longest chain
layers = c.layers()         # ranked layering of the poset

c.is_chain(list(chain))     # True
c.is_antichain([0, 1, 2])   # depends on the set
```

A chain is a set of mutually comparable elements; an antichain is a set of mutually
incomparable ones. `is_chain` and `is_antichain` check any list of indices you pass.

## Past, future, intervals

```python
x, y = 10, 400

past = c.past(x)            # indices i with i < x
future = c.future(x)        # indices j with x < j
interval = c.interval(x, y) # indices in future(x) ∩ past(y)
```

`interval(x, y)` is the Alexandrov interval, the discrete version of the set of points
between two spacetime events.

## Relation fraction

```python
r = c.relation_fraction()   # |relations| / max possible, in [0, 1]
```

A total order (a chain) has fraction 1; an antichain has fraction 0. A causet
sprinkled into a $d$-dimensional Minkowski diamond has a fraction characteristic of
$d$ — that is what the next method exploits.

## Estimating the dimension

Myrheim-Meyer inverts the relation fraction to estimate the dimension of the
spacetime a causet was sprinkled into, from the order alone:

```python
for d in (2, 3):
    st = pc.spacetime.MinkowskiDiamond(dimension=d)
    c = pc.causet(n=300, spacetime=st, seed=42)
    print(f"built in {d}D -> fraction {c.relation_fraction():.3f}, "
          f"estimate {c.myrheim_meyer_dimension():.2f}")
```

At n=300 this prints roughly `1.96` for 2D and `2.88` for 3D. The estimate gets
tighter as n grows. This is the discrete version of "measuring the dimension of
spacetime", the kind of quantity causal set research reads off the order.

## Entanglement entropy

Entanglement entropy lives on the field side, not the set:

```python
phi = pc.field("scalar", mass=1.0)
Q = phi.on(c)
S = Q.entanglement_entropy([0, 1, 2, 3, 4])   # region = list of indices
```

See [[guides/Field Theory|Field Theory]] for the conventions.

---

See [[guides/Causal Sets|Causal Sets]] for making the set in the first place, and
[[guides/Tutorials|Tutorials]] for these methods used end to end.
