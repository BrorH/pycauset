# pycauset.vis

The `pycauset.vis` module provides tools for visualizing Causal Sets.

## Functions

*   [[docs/pycauset.vis/plot_embedding.md|pycauset.vis.plot_embedding]]: Plot the spacetime embedding of a causal set.
*   [[docs/pycauset.vis/plot_hasse.md|pycauset.vis.plot_hasse]]: Generate a Hasse diagram of a causal set.
*   [[docs/pycauset.vis/plot_causal_matrix.md|pycauset.vis.plot_causal_matrix]]: Visualize a causal matrix as a heatmap.

## Description

This module leverages [Plotly](https://plotly.com/python/) to create interactive 3D visualizations. It is designed to handle large causal sets efficiently by using smart sampling and on-demand coordinate generation.

## Examples

```python
from pycauset import CausalSet
from pycauset.vis import plot_embedding

c = CausalSet(n=1000)
fig = plot_embedding(c)
fig.show()
```
