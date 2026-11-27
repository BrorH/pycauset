# pycauset.vis

The `pycauset.vis` module provides tools for visualizing Causal Sets.

## Functions

*   [[docs/pycauset.vis/plot_embedding.md|pycauset.vis.plot_embedding]]: Plot the spacetime embedding of a causal set.
*   [[docs/pycauset.vis/plot_hasse.md|pycauset.vis.plot_hasse]]: Generate a Hasse diagram of a causal set.
*   [[docs/pycauset.vis/plot_causal_matrix.md|pycauset.vis.plot_causal_matrix]]: Visualize a causal matrix as a heatmap.

## Call surface (R2)

The same plotters are also reachable as `CausalSet` methods (`c.plot_embedding()`,
`c.plot_hasse()`, `c.plot_causal_matrix()`) and as lazy top-level aliases
(`pc.plot_embedding`, `pc.plot_hasse`, `pc.plot_causal_matrix`) plus the one-verb
[[docs/functions/pycauset.show.md|pc.show]]. Above `max_points` they draw a seeded
subset and emit a `PyCausetPerformanceWarning`; `force=True` renders everything.

## Description

This module uses [Plotly](https://plotly.com/python/) for interactive 3D plots. Large causal sets are handled by sampling and by generating coordinates on demand.

## Examples

```python
from pycauset import CausalSet
from pycauset.vis import plot_embedding

c = CausalSet(n=1000)
fig = plot_embedding(c)
fig.show()
```
