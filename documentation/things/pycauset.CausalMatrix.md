
```
pycauset.CausalMatrix(N, saveas=None, populate=True, seed=None)
```
Create a causal matrix of size $N\times N$.

Inherits from [[pycauset.Matrix]].

### Parameters:
- N: int. The dimension of the causal set.
- saveas: str (_optional_). Name of the binary file where the causal matrix is stored in ```.pycauset/```. Can be a path to specify custom storage location. If `None`, the matrix will not be saved (unless [[pycauset.save]] is set to `True`).
- populate: bool (_optional_). If `True`, the matrix will automatically be filled with random bits. For specifics see [[RNG]]. If `False`, the matrix will be initialized as empty.
- seed: int (_optional_). Overrides [[pycauset.seed]] for this call only. Use it to make random populations deterministic.

### Returns:
[[pycauset.CausalMatrix]] instance

