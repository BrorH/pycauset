# pycauset.vis.plot_hasse

```python
pycauset.vis.plot_hasse(
    causet: CausalSet,
    max_points: int = 500,
    force: bool = False,
    title: str = None,
    marker_size: int = 5,
    line_width: int = 1,
    line_color: str = 'rgba(255, 255, 255, 0.3)'
) -> plotly.graph_objects.Figure
```

Generates a Hasse diagram of the Causal Set.

## Parameters

*   **causet** (*CausalSet*): The causal set to visualize. It needs coordinates: a sprinkled
    causal set (a causet built from a matrix has none, `ValueError`).
*   **max_points** (*int*, optional): Draw at most this many elements. Above it, a seeded random
    subset is drawn with a `PyCausetPerformanceWarning`, and its lines are the links of that
    sub-order (two sampled elements with no *sampled* element between them), not all links of
    the causal set; the title says so. Defaults to 500.
*   **force** (*bool*, optional): Draw every element, ignoring `max_points`.
*   **title** (*str*, optional): The title of the plot.
*   **marker_size** (*int*, optional): Size of the element markers. Defaults to 5.
*   **line_width** (*int*, optional): Width of the link lines. Defaults to 1.
*   **line_color** (*str*, optional): Color of the link lines. Defaults to 'rgba(255, 255, 255, 0.3)'.

## Returns

*   **plotly.graph_objects.Figure**: A Plotly figure object containing the Hasse diagram.

## Raises

*   **ValueError**: the causal set has no coordinates (it was built from a matrix).
*   **ImportError**: Plotly is not installed (`pip install "pycauset[plot]"`).

## Description

A Hasse diagram displays the transitive reduction of the partial order. Elements are placed at their spacetime coordinates, and lines are drawn only between immediate causal neighbors (links). This reveals the "skeleton" of the causal structure.

Elements are drawn at their coordinates. Embeddings beyond 3 dimensions are shown as their
first three axes, with a `PyCausetWarning`.

## See Also

*   [[guides/Visualization|Visualization Guide]]: For a guide on visualizing causal sets.
*   [[docs/pycauset.vis/plot_embedding.md|pycauset.vis.plot_embedding]]: For visualizing the embedding without links.
