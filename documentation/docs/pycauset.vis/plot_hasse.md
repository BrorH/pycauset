# pycauset.vis.plot_hasse

```python
pycauset.vis.plot_hasse(
    causet: CausalSet,
    title: str = None,
    marker_size: int = 5,
    line_width: int = 1,
    line_color: str = 'rgba(255, 255, 255, 0.3)'
) -> plotly.graph_objects.Figure
```

Generates a Hasse diagram of the Causal Set.

## Parameters

*   **causet** (*CausalSet*): The causal set to visualize.
*   **title** (*str*, optional): The title of the plot.
*   **marker_size** (*int*, optional): Size of the element markers. Defaults to 5.
*   **line_width** (*int*, optional): Width of the link lines. Defaults to 1.
*   **line_color** (*str*, optional): Color of the link lines. Defaults to 'rgba(255, 255, 255, 0.3)'.

## Returns

*   **plotly.graph_objects.Figure**: A Plotly figure object containing the Hasse diagram.

## Raises

*   **ValueError**: If the causal set is too large (> 500 elements) for a Hasse diagram.

## Description

A Hasse diagram displays the transitive reduction of the partial order. Elements are placed at their spacetime coordinates, and lines are drawn only between immediate causal neighbors (links). This reveals the "skeleton" of the causal structure.

The function automatically applies coordinate transformations for supported spacetimes (e.g., `MinkowskiDiamond`, `MinkowskiCylinder`) to improve readability.

## See Also

*   [[guides/Visualization|Visualization Guide]]: For a comprehensive guide on visualizing causal sets.
*   [[docs/pycauset.vis/plot_embedding.md|pycauset.vis.plot_embedding]]: For visualizing the embedding without links.
