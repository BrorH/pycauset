# pycauset.causet

```python
pycauset.causet(*, n=None, density=None, spacetime=None, seed=None, matrix=None) -> CausalSet
```

A causal set: sprinkled (`n` or `density`, `spacetime`, `seed`) into a [[docs/classes/spacetime/pycauset.spacetime.MinkowskiDiamond.md|MinkowskiDiamond]] (the 2D diamond by default), or built from a causal matrix (`matrix`). Same as [[docs/classes/spacetime/pycauset.CausalSet.md|pycauset.CausalSet]].

## Parameters

See [[docs/classes/spacetime/pycauset.CausalSet.md|pycauset.CausalSet]] for parameter details.

## Returns

*   **CausalSet**

## Example

```python
import pycauset

c = pycauset.causet(n=1000, seed=42)
```
