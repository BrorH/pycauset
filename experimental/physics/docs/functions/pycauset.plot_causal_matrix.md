# pycauset.plot_causal_matrix

```python
pycauset.plot_causal_matrix(causet, **kwargs)
```

Plot a causal set's causal matrix as a heatmap.

Lazy top-level sugar for the `CausalSet.plot_causal_matrix` method, which is the
primary citizen. Plotly is imported on first use.

## Parameters

*   **causet** (*CausalSet*): The causal set to plot.
*   **kwargs**: Passed through to `CausalSet.plot_causal_matrix`.

## Returns

*   Plotly `Figure`.
