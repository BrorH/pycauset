# pycauset.vis.plot_embedding

```python
pycauset.vis.plot_embedding(
    causet: CausalSet,
    max_points: int = 50000,
    force: bool = False,
    title: str = None,
    marker_size: int = 2
) -> plotly.graph_objects.Figure
```

Visualize the spacetime embedding of a causal set.

## Parameters

*   **causet** (*CausalSet*): The causal set to visualize.
*   **max_points** (*int*, optional): Render at most this many elements. Above it, a seeded random subset is drawn and a `PyCausetPerformanceWarning` is emitted. Defaults to 50000.
*   **force** (*bool*, optional): Render every element, ignoring `max_points`. Defaults to `False`.
*   **title** (*str*, optional): The title of the plot. If `None`, a default title is generated.
*   **marker_size** (*int*, optional): The size of the scatter points. Defaults to 2.

## Returns

*   **plotly.graph_objects.Figure**: A Plotly figure object containing the scatter plot.

## Description

The plot reads the spacetime's authored `to_embedding` / `boundary` / `display_axes`
declarations to draw the shape; a geometry-free custom spacetime renders its raw
coordinates with generic axis labels. Embedding dimensions beyond 3 are shown as the
first three axes with an explicit warning. Points are colored by their time
coordinate.

## See Also

*   [[guides/Visualization|Visualization Guide]]: For a guide on visualizing causal sets.
*   [[docs/pycauset.vis/plot_hasse.md|pycauset.vis.plot_hasse]]: For the Hasse diagram.
*   [[docs/pycauset.vis/plot_causal_matrix.md|pycauset.vis.plot_causal_matrix]]: For the causal-matrix heatmap.
