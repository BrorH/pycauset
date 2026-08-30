# pycauset.plot_hasse

```python
pycauset.plot_hasse(causet, **kwargs)
```

Plot a causal set's Hasse diagram (the transitive reduction of its order).

Lazy top-level sugar for the `CausalSet.plot_hasse` method, which is the primary
citizen. Elements are placed at their spacetime coordinates and lines connect only
immediate causal neighbors (links). Plotly is imported on first use.

## Parameters

*   **causet** (*CausalSet*): The causal set to plot.
*   **kwargs**: Passed through to `CausalSet.plot_hasse`.

## Returns

*   Plotly `Figure`.
