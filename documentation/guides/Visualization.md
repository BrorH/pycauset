# Visualization Guide

PyCauset ships Plotly-based plotters for causal sets. You get three views: the
spacetime embedding, the Hasse diagram, and the causal-matrix heatmap.

## Basic usage

The plotters live at the top level and as methods on `CausalSet`. Both do the same
thing:

```python
import pycauset as pc

c = pc.causet(n=3000, seed=42)

pc.plot_embedding(c).show()   # top-level function
c.plot_embedding().show()     # method (no import needed)
pc.show(c)                    # one-verb sugar: plot + .show()
```

`pc.show(c)` picks the embedding plot and calls `.show()` for you. The other two
views are `plot_hasse` and `plot_causal_matrix`, also available both ways.

Plotting needs `plotly`, an optional extra: `pip install "pycauset[plot]"`. Without it the
plotters raise an `ImportError` that says so.

Here is what the embedding plot of a 2D Minkowski diamond looks like:

![Minkowski diamond embedding](../docs/assets/gallery/diamond_embedding.png)

*3000 points in a 2D diamond. Time runs up the vertical axis; colour is the time
coordinate, and the white outline is the diamond boundary.*

## Reproducibility

The causal set itself is reproducible when you pass a `seed`. The large-set
subsampling uses a fixed internal seed too, so plotting the same causal set always
shows the same subset.

```python
c = pc.causet(n=5_000, seed=12345)
c.plot_hasse().show()   # 500 of the 5,000 elements, the same 500 every time
```

## Large sets

Rendering millions of points in a browser is not practical, so each plotter caps the
number of points it draws. Above the cap it draws a seeded random subset and emits a
`PyCausetPerformanceWarning` naming what was sampled. Pass `force=True` (or
`max_points=None`) to render everything.

| Plotter | Default cap (`max_points`) |
| :--- | :--- |
| `plot_embedding` | 50,000 |
| `plot_hasse` | 500 |
| `plot_causal_matrix` | 2,000 |

```python
c.plot_hasse()                   # warns + subsets above 500
c.plot_hasse(max_points=200)     # a smaller subset
c.plot_embedding(force=True)     # every point (here under the cap anyway)
```

## Customizing the plot

```python
c.plot_embedding(
    title="My universe",
    marker_size=3,
)
```

Each plotter returns a plain Plotly `Figure`, so anything you can do to a Plotly
figure you can do here (change the template, add traces, save to file).

## Coordinates and boundaries

The plotter does not guess the shape of a spacetime; the spacetime declares it.
`MinkowskiDiamond` gives its outline (`boundary()`: the diamond in 2D, its rim circle and
tip-to-tip lines in 3D) and its axis names (`display_axes()`: `t`, `x`, `y`, ...). Time is
the vertical axis.

Lightweight by design: three plots and `show`. Causal-set visualization gets a release of
its own later.

Embeddings with more than 3 dimensions show the first three
axes and warn explicitly; nothing is silently truncated.

![4D diamond, first three axes](../docs/assets/gallery/diamond_4d_embedding.png)

*A 4D diamond has no flat 2D picture; the plotter renders the first three axes and
warns, rather than silently dropping the fourth.*

## Hasse diagrams

`plot_hasse` draws only the links (the transitive reduction), so you see the skeleton
of the partial order rather than every relation. It places elements at their
spacetime coordinates.

```python
c = pc.causet(n=80, seed=7)
c.plot_hasse().show()
```

![Hasse diagram](../docs/assets/gallery/diamond_hasse.png)

*Only the links are drawn, so the skeleton of the partial order is visible. Lines run
between immediate causal neighbours, placed at their spacetime coordinates.*

The default cap is 500 points; above that it subsets with a warning, and then its lines are
the links of the sampled sub-order (two sampled elements with no sampled element between
them), not all links of the causal set: the title and the warning say so.

The embedding and Hasse plots need coordinates, so they work for sprinkled causal sets. A causal
set built from a matrix (`CausalSet(n, matrix=M)`, `pc.synthetic`) has none and raises
`ValueError`; `plot_causal_matrix` plots any causal set.

## Causal-matrix heatmaps

`plot_causal_matrix` draws the causal matrix $C$ as a heatmap. Since the matrix is
strictly upper triangular for a sorted causal set, you see a triangular pattern.

```python
c.plot_causal_matrix(color_scale="Greys").show()
```

![Causal matrix heatmap](../docs/assets/gallery/causal_matrix.png)

*The strictly upper-triangular pattern is the signature of a time-labelled causal set:
elements below the diagonal are never in the past of anything above it.*

The default cap is 2,000 points; above that it subsets with a warning.

See [[docs/pycauset.vis/index.md|pycauset.vis]] for the exact signatures.
