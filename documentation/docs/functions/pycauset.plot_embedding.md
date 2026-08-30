# pycauset.plot_embedding

```python
pycauset.plot_embedding(causet, **kwargs)
```

Plot a causal set's embedding (spacetime coordinates).

Lazy top-level sugar for [[pycauset.CausalSet.plot_embedding|CausalSet.plot_embedding]];
the method on `CausalSet` is the primary citizen and accepts the same keyword
arguments (subsetting, `force`, title, etc.). Plotly is imported on first use.

## Parameters

*   **causet** (*CausalSet*): The causal set to plot.
*   **kwargs**: Passed through to `CausalSet.plot_embedding`.

## Returns

*   Plotly `Figure`.
