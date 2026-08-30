# pycauset.plot_embedding

```python
pycauset.plot_embedding(causet, **kwargs)
```

Plot a causal set's embedding (spacetime coordinates).

Lazy top-level sugar for the `CausalSet.plot_embedding` method, which is the primary
citizen and accepts the same keyword arguments (subsetting, `force`, title, etc.).
Plotly is imported on first use.

## Parameters

*   **causet** (*CausalSet*): The causal set to plot.
*   **kwargs**: Passed through to `CausalSet.plot_embedding`.

## Returns

*   Plotly `Figure`.
