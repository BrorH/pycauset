# pycauset.plot_causal_matrix

```python
pycauset.plot_causal_matrix(causet, **kwargs)
```

Plot a causal set's causal matrix as a heatmap.

Lazy top-level sugar for
[[pycauset.CausalSet.plot_causal_matrix|CausalSet.plot_causal_matrix]]; the method
on `CausalSet` is the primary citizen. Plotly is imported on first use.

## Parameters

*   **causet** (*CausalSet*): The causal set to plot.
*   **kwargs**: Passed through to `CausalSet.plot_causal_matrix`.

## Returns

*   Plotly `Figure`.
