# pycauset.causet

```python
pycauset.causet(*, n=None, density=None, spacetime=None, seed=None, matrix=None) -> CausalSet
```

Lower-case convenience factory that returns a [[pycauset.CausalSet]].

## Parameters

See [[pycauset.CausalSet]] for parameter details.

## Returns

*   **CausalSet**: A new instance of [[pycauset.CausalSet]].

## Example

```python
import pycauset

c = pycauset.causet(n=1000, seed=42)
```
