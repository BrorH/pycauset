# pycauset.vis.plot_embedding

```python
pycauset.vis.plot_embedding(
    causet: CausalSet,
    sample_size: int = 50000,
    title: str = None,
    marker_size: int = 2
) -> plotly.graph_objects.Figure
```

Generates an interactive 3D (or 2D) scatter plot of the Causal Set embedding.

## Parameters

*   **causet** (*CausalSet*): The causal set to visualize.
*   **sample_size** (*int*, optional): The maximum number of points to display. If `causet.n` > `sample_size`, a random subset is shown to maintain performance. Defaults to 50000.
*   **title** (*str*, optional): The title of the plot. If None, a default title is generated.
*   **marker_size** (*int*, optional): The size of the scatter points. Defaults to 2.

## Returns

*   **plotly.graph_objects.Figure**: A Plotly figure object containing the scatter plot.

## Description

This function visualizes the causal set by regenerating the spacetime coordinates of its elements. It uses the `make_coordinates` backend to efficiently retrieve positions without storing them permanently.

For large causal sets, the function automatically downsamples the points to `sample_size` to ensure the visualization remains responsive. The points are colored according to their time coordinate.

## See Also

*   [[guides/Visualization|Visualization Guide]]: For a comprehensive guide on visualizing causal sets.
*   `pycauset.CausalSet.coordinates`: Method on [[docs/classes/spacetime/pycauset.CausalSet.md|pycauset.CausalSet]] used to retrieve coordinates.
