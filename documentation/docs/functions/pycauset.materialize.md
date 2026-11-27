# pycauset.materialize

```python
pycauset.materialize(a)
```

Returns `a` with its elements stored: a lazy result (such as `C + C.T` of a causal matrix)
is evaluated, a scaled, transposed or conjugated view is written out. A plain stored matrix
is returned as is. Values never change; only where they come from.

Operations read views and lazy results directly, so this is only worth calling to trade
memory for repeated-access speed, for example before millions of single-element reads.
See [[guides/Lazy Evaluation|Lazy Evaluation]].

## Example

```python
import pycauset as pc

C = pc.causal_matrix(100, seed=1)
D = C - C.T             # lazy: read from C's bits
E = pc.materialize(D)   # same values, stored
assert not E.is_lazy
```
