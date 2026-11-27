# pycauset.show

```python
pycauset.show(causet) -> None
```

One-verb sugar: plot a causal set's embedding and open it in the browser (equivalent to `causet.plot_embedding().show()`).

## Example

```python
import pycauset as pc

pc.show(pc.causet(n=3000, seed=42))
```

## See also

- [[docs/classes/spacetime/pycauset.CausalSet.md|CausalSet]]
- [[docs/pycauset.vis/plot_embedding.md|plot_embedding]]
- [[guides/Visualization.md|Visualization guide]]
