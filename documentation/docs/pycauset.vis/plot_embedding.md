# pycauset.vis.plot_embedding

```python
pycauset.vis.plot_embedding(
    causet: CausalSet,
    max_points: int = 50000,
    force: bool = False,
    title: str = None,
    marker_size: int = 2,
    show_relations: bool = False,
    max_relations: int = 20000
) -> plotly.graph_objects.Figure
```

Visualize the spacetime embedding of a causal set.

## Parameters

*   **causet** (*CausalSet*): The causal set to visualize.
*   **max_points** (*int*, optional): Render at most this many elements. Above it, a seeded random subset is drawn and a `PyCausetPerformanceWarning` is emitted. Defaults to 50000.
*   **force** (*bool*, optional): Render every element, ignoring `max_points`. Defaults to `False`.
*   **title** (*str*, optional): The title of the plot. If `None`, a default title is generated.
*   **marker_size** (*int*, optional): The size of the scatter points. Defaults to 2.
*   **show_relations** (*bool*, optional): Draw a faint line for every causal pair `a < b`
    among the plotted points (the whole relation, not just the links). The pairs come from the
    points' causal sub-matrix, selected in the engine; above 5,000 plotted points they are drawn
    among a seeded random 5,000 of them, with a warning.
*   **max_relations** (*int*, optional): Draw at most this many pairs (a seeded random subset
    beyond, with a warning). Defaults to 20000.

The causal set needs coordinates: a sprinkled one. A causal set built from a matrix has none
(`ValueError`); plot its matrix with `plot_causal_matrix`.

## Returns

*   **plotly.graph_objects.Figure**: A Plotly figure object containing the scatter plot.

## Description

The plot reads the spacetime's `boundary()` and `display_axes()` to draw its outline and
name the axes. Embedding dimensions beyond 3 are shown as the
first three axes with an explicit warning. Points are colored by their time
coordinate.

## See Also

*   [[guides/Visualization|Visualization Guide]]: For a guide on visualizing causal sets.
*   [[docs/pycauset.vis/plot_hasse.md|pycauset.vis.plot_hasse]]: For the Hasse diagram.
*   [[docs/pycauset.vis/plot_causal_matrix.md|pycauset.vis.plot_causal_matrix]]: For the causal-matrix heatmap.
