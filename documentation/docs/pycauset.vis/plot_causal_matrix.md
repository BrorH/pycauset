# pycauset.vis.plot_causal_matrix

```python
pycauset.vis.plot_causal_matrix(
    causet: CausalSet,
    title: str = None,
    color_scale: str = 'Greys'
) -> plotly.graph_objects.Figure
```

Visualize the Causal Matrix (Adjacency Matrix) as a heatmap.

## Parameters

*   **causet** (*CausalSet*): The causal set to visualize.
*   **title** (*str*, optional): The title of the plot.
*   **color_scale** (*str*, optional): The color scale to use (e.g., 'Greys', 'Viridis'). Defaults to 'Greys'.

## Returns

*   **plotly.graph_objects.Figure**: A Plotly figure object containing the heatmap.

## Raises

*   **ValueError**: If the matrix is too large (> 2000 elements) to render effectively.

## Description

This function visualizes the Causal Matrix (Adjacency Matrix) $C$ as a heatmap. Since the matrix is strictly upper triangular (for a sorted causal set), the heatmap will show a triangular pattern. This is useful for inspecting the density and structure of causal relations.

## See Also

*   [[guides/Visualization|Visualization Guide]]: For a comprehensive guide on visualizing causal sets.
*   [[docs/pycauset.vis/plot_hasse.md|pycauset.vis.plot_hasse]]: For visualizing the causal structure as a graph.
