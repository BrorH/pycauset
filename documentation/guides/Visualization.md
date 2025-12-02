# Visualization Guide

PyCauset provides built-in tools to visualize Causal Sets embedded in spacetime. This guide explains how to use the [[pycauset.vis]] module to create interactive 3D plots.

## Overview

Visualizing high-dimensional causal structures is crucial for understanding their geometry and topology. PyCauset uses **Plotly** to generate interactive figures that allow you to rotate, zoom, and inspect the causal set.

## Basic Usage

The primary function for visualization is [[pycauset.vis.plot_embedding]].

```python
from pycauset import CausalSet
from pycauset.vis import plot_embedding

# 1. Generate a Causal Set
# Use a seed for reproducibility!
c = CausalSet(n=5000, density=100, seed=42)

# 2. Create the plot
fig = plot_embedding(c)

# 3. Display it
fig.show()
```

## Reproducibility

To ensure your visualization is identical every time, you must ensure the [[pycauset.CausalSet]] itself is reproducible (using `seed` in `CausalSet()`). The visualization function uses a fixed internal seed for subsampling, so plotting the same `CausalSet` object will always yield the same plot.

```python
# 1. Reproducible Causal Set
c = CausalSet(n=100_000, seed=12345)

# 2. Plotting
# This will always show the same subset of points
fig = plot_embedding(c)
```

## Handling Large Sets

Visualizing millions of points in a web browser is not feasible. PyCauset solves this by **Smart Sampling**.

When you pass a large [[pycauset.CausalSet]] to `plot_embedding`, it automatically samples a subset of points (default 50,000) to display. This preserves the global structure while keeping the visualization responsive.

```python
# A very large set (1 million elements)
c_huge = CausalSet(n=1_000_000)

# This will plot a random sample of 50,000 points
fig = plot_embedding(c_huge)
fig.show()
```

You can control the sample size with the `sample_size` parameter:

```python
fig = plot_embedding(c_huge, sample_size=10000)
```

## Coordinate Regeneration

Under the hood, PyCauset does not store the coordinates of every element in memory. Instead, it uses the **Sprinkler** algorithm to regenerate coordinates on-demand using the original random seed.

This means you can visualize a subset of a massive causal set without ever generating the full coordinate table, saving gigabytes of RAM.

See [[pycauset.CausalSet.coordinates]] for more details on the coordinate system.

## Customizing the Plot

You can customize the plot title and marker size.

```python
fig = plot_embedding(
    c, 
    title="My Universe", # Custom title
    marker_size=3        # Larger points
)
```

Since [[pycauset.vis.plot_embedding]] returns a Plotly figure, you can modify it how you would any Plotly figure.

## Coordinate Transformations & Boundaries

The visualization module automatically handles coordinate transformations for specific spacetimes to make them easier to interpret:

*   **MinkowskiDiamond (2D)**: Lightcone coordinates $(u, v)$ are rotated to Cartesian coordinates $(t, x)$. The diamond boundary is drawn in white.
*   **MinkowskiCylinder (2D)**: Coordinates are mapped to a 3D cylinder visualization. The top and bottom rings of the cylinder are drawn in white.
*   **MinkowskiBox (2D)**: Coordinates are displayed as Cartesian $(t, x)$. The rectangular boundary is drawn in white.

## Hasse Diagrams

[[pycauset.vis.plot_hasse]] generates a Hasse diagram, which visualizes the causal structure (partial order) directly.

```python
from pycauset.vis import plot_hasse

fig = plot_hasse(c, title="Hasse Diagram")
fig.show()
```

A Hasse diagram draws elements at their spacetime coordinates but draws lines (links) only between immediate causal neighbors. This reveals the "skeleton" of the causal structure.

*   **Note**: Hasse diagrams are computationally expensive and visually cluttered for large sets. The function will raise an error if $N > 500$.

## Causal Matrix Heatmaps

`plot_causal_matrix` visualizes the Causal Matrix (Adjacency Matrix) $C$ as a heatmap.

```python
from pycauset.vis import plot_causal_matrix

fig = plot_causal_matrix(c, title="Causal Matrix")
fig.show()
```

Since the causal matrix is strictly upper triangular (for a sorted causal set), the heatmap will show a triangular pattern. This is useful for inspecting the density and structure of causal relations.

*   **Note**: For very large sets ($N > 2000$), this plot may become slow to render.

## Dependencies

Visualization requires the `plotly` library.

```bash
pip install plotly
```

If `plotly` is not installed, importing `pycauset.vis` will raise an `ImportError`.
