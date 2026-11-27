# pycauset.synthetic

Synthetic poset generators ("a causet is just a poset"). Each builds a valid causal order
directly (no continuum geometry) and returns a `CausalSet` that passes `validate()`.

```python
from pycauset import synthetic

c = synthetic.chain(100)            # the total order
c = synthetic.antichain(100)        # the empty order
c = synthetic.transitive_percolation(0.4, 100, seed=1)
c = synthetic.random_dag_order(0.4, 100, seed=1)
c = synthetic.product_order((2, 3)) # a grid poset
c = synthetic.poset([(0, 1), (1, 2)])
```

## Generators

| function | description |
| :-- | :-- |
| `chain(n)` | The total order `0 < 1 < … < n-1`. |
| `antichain(n)` | The empty order (no relations). |
| `transitive_percolation(p, n, seed=None)` | Random causet from bond percolation on a total order. |
| `random_dag_order(p, n, seed=None)` | Random acyclic upper-triangular edges + transitive closure. |
| `product_order(dims)` | The grid poset (product of chains). |
| `poset(relations, n=None)` | An explicit user order from `(i, j)` pairs, transitively closed. |

Each generator applies the transitive closure internally and returns a validated `CausalSet`.

## See also

- [[docs/classes/spacetime/pycauset.CausalSet.md|CausalSet]]
- [[guides/Causal Sets.md|Causal Sets guide]]
- [[project/plans/R2_SPACETIME_LIBRARY.md|R2 Spacetime Library]]
