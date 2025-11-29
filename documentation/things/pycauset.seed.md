```
pycauset.seed
```
Global integer seed for random number generation.

- Type: `int` or `None`
- Default: `None`

If set, this seed is used for all subsequent random matrix generations (e.g., via [[pycauset.CausalMatrix]] with `populate=True` or `pycauset.CausalMatrix.random`), ensuring reproducibility.
