# pycauset.vis.plot_causal_matrix

```python
pycauset.vis.plot_causal_matrix(
    causet: CausalSet,
    max_points: int = 2000,
    force: bool = False,
    title: str = None,
    color_scale: str = None
) -> plotly.graph_objects.Figure
```

Visualize the causal matrix as a heatmap.

## Parameters

*   **causet** (*CausalSet*): The causal set to visualize.
*   **max_points** (*int*, optional): Render at most this many elements. Above it, a seeded random subset is drawn and a `PyCausetPerformanceWarning` is emitted. Defaults to 2000.
*   **force** (*bool*, optional): Render every element, ignoring `max_points`. Defaults to `False`.
*   **title** (*str*, optional): The title of the plot.
*   **color_scale** (*str*, optional): Any Plotly continuous colorscale name (e.g. `'Greys'`, `'Viridis'`). Defaults to a two-tone scale (dark `0`, teal `1`) that reads as a boolean image.

## Returns

*   **plotly.graph_objects.Figure**: A Plotly figure object containing the heatmap.

## Description

The causal matrix $C$ is boolean: a bright cell means element $i$ is in the causal
past of element $j$. For a causal set whose elements are labelled by time the matrix
is strictly upper triangular, so the heatmap shows a crisp triangle. This is useful
for inspecting the density and structure of causal relations.

## See Also

*   [[guides/Visualization|Visualization Guide]]: For a guide on visualizing causal sets.
*   [[docs/pycauset.vis/plot_hasse.md|pycauset.vis.plot_hasse]]: For visualizing the causal structure as a graph.
