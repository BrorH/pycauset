```
pycauset.seed
```

Type: Optional[int]
Default: None

Effect: Sets the seed used by random population routines (`populate=True` constructors and `pycauset.CausalMatrix.random`). Assign an integer for deterministic fills or `None` to return to non-deterministic behavior.