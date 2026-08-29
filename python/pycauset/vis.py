try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    go = None
    px = None

import random
import warnings
from typing import Optional

import numpy as np

from ._internal.warnings import PyCausetPerformanceWarning
from .causet import CausalSet


def _subset(n: int, max_points: Optional[int], force: bool, kind: str):
    """Return (indices_or_None, note) for the large-set policy (R2_VIZ).

    Above ``max_points`` a seeded random subset is drawn and a
    `PyCausetPerformanceWarning` is emitted; ``force=True`` (or ``max_points=None``)
    renders everything.
    """
    if max_points is None or force or n <= max_points:
        return None, ""
    rng = random.Random(42)
    indices = sorted(rng.sample(range(n), max_points))
    note = f" (Subsample of {max_points} points)"
    warnings.warn(
        f"{kind} of a {n}-element causal set: plotting a seeded random subset of "
        f"{max_points} elements. Pass force=True (or max_points=None) to render all.",
        PyCausetPerformanceWarning,
        stacklevel=3,
    )
    return indices, note


def _check_plotly():
    if go is None:
        raise ImportError("Plotly is required for visualization. Install it with 'pip install plotly'.")


def _spacetime_of(causet: CausalSet):
    """The causet's spacetime (or None if unavailable)."""
    return getattr(causet, "_spacetime", None)


def _embedding(coords, st):
    """Apply the spacetime's authored embedding (default: identity).

    Prefers the `Spacetime.to_embedding` hook; falls back to the native
    `transform_coordinates` alias. If neither is present (geometry-free custom
    spacetime), the raw coordinates are used, never inferred.
    """
    coords = np.asarray(coords, dtype=float)
    if st is None:
        return coords
    fn = getattr(st, "to_embedding", None) or getattr(st, "transform_coordinates", None)
    if fn is None:
        return coords
    return np.asarray(fn(coords), dtype=float)


def _boundary_paths(st):
    """Boundary paths in embedding coordinates (default: none).

    Prefers the `Spacetime.boundary` hook; falls back to the native `get_boundary`
    alias. The declared paths are already in display (embedding) coordinates.
    """
    if st is None:
        return []
    fn = getattr(st, "boundary", None) or getattr(st, "get_boundary", None)
    if fn is None:
        return []
    paths = fn() or []
    return [np.asarray(p, dtype=float) for p in paths]


def _axis_labels(st, dim: int):
    """Axis labels for the embedding (default: generic ``c0, c1, …``)."""
    if st is not None:
        fn = getattr(st, "display_axes", None)
        if fn is not None:
            labels = fn()
            if labels:
                labels = [str(x) for x in labels]
                if len(labels) >= dim:
                    return labels[:dim]
    return [f"c{i}" for i in range(dim)]


def _boundary_traces(paths, dim: int):
    """Plotly traces for boundary paths already in embedding coordinates."""
    traces = []
    for b in paths:
        b = np.asarray(b, dtype=float)
        if b.ndim != 2 or b.shape[1] < dim:
            continue
        if dim == 2:
            traces.append(go.Scatter(
                x=b[:, 1], y=b[:, 0],
                mode="lines",
                line=dict(color="white", width=2),
                name="Boundary",
                hoverinfo="skip",
            ))
        elif dim >= 3:
            traces.append(go.Scatter3d(
                x=b[:, 1], y=b[:, 2], z=b[:, 0],
                mode="lines",
                line=dict(color="white", width=2),
                name="Boundary",
                hoverinfo="skip",
            ))
    return traces


def plot_embedding(
    causet: CausalSet,
    max_points: int = 50000,
    sample_size: Optional[int] = None,
    force: bool = False,
    title: Optional[str] = None,
    marker_size: int = 2
):
    """Visualize the spacetime embedding of the Causal Set.

    The spacetime's authored ``to_embedding`` / ``boundary`` / ``display_axes``
    declarations drive the plot; a geometry-free custom spacetime renders raw with
    generic axis labels (never inferred). Embedding dimensions beyond 3 are shown
    as the first three axes with an explicit warning (never silently truncated).
    """
    _check_plotly()

    if sample_size is not None:
        max_points = sample_size

    n = causet.n
    indices, note = _subset(n, max_points, force, "plot_embedding")

    st = _spacetime_of(causet)
    coords = _embedding(causet.coordinates(indices=indices), st)
    dim = coords.shape[1]
    labels = _axis_labels(st, dim)

    if dim > 3:
        warnings.warn(
            f"plot_embedding: embedding has {dim} dimensions; rendering the first 3 "
            f"axes ({labels[:3]}) and dropping {labels[3:]}.",
            UserWarning,
            stacklevel=2,
        )
        coords = coords[:, :3]
        labels = labels[:3]
        dim = 3

    boundary_traces = _boundary_traces(_boundary_paths(st), dim)

    # Embedding contract: column 0 is time (vertical), columns 1.. are spatial.
    if dim == 2:
        data = [go.Scatter(
            x=coords[:, 1],
            y=coords[:, 0],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=coords[:, 0],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Time"),
            ),
            name="Events",
        )] + boundary_traces
        fig = go.Figure(data=data)
        fig.update_layout(
            title=title or f"2D Spacetime Embedding{note}",
            xaxis_title=labels[1],
            yaxis_title=labels[0],
            template="plotly_dark",
            showlegend=False,
        )
    else:  # dim == 3
        x_data = coords[:, 1]
        y_data = coords[:, 2]
        z_data = coords[:, 0]
        data = [go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=z_data,
                colorscale="Viridis",
                opacity=0.8,
                showscale=True,
                colorbar=dict(title="Time"),
            ),
            name="Events",
        )] + boundary_traces
        fig = go.Figure(data=data)
        fig.update_layout(
            title=title or f"{dim}D Spacetime Embedding{note}",
            scene=dict(
                xaxis_title=labels[1],
                yaxis_title=labels[2],
                zaxis_title=labels[0],
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False,
        )

    return fig


def plot_hasse(
    causet: CausalSet,
    max_points: int = 500,
    force: bool = False,
    title: Optional[str] = None,
    marker_size: int = 5,
    line_width: int = 1,
    line_color: str = 'rgba(255, 255, 255, 0.3)'
):
    """Generate a Hasse diagram of the Causal Set.

    A Hasse diagram displays the transitive reduction of the partial order.
    Elements are placed at their spacetime coordinates, and lines are drawn only
    between immediate causal neighbors (links). Uses the spacetime's authored
    embedding; dimensions beyond 3 are shown as the first three axes with a warning.
    """
    _check_plotly()

    indices, _ = _subset(causet.n, max_points, force, "plot_hasse")

    if indices is not None:
        coords = causet.coordinates(indices=indices)
        C_dense = np.array(causet.C, dtype=int)[np.ix_(indices, indices)]
        node_ids = indices
    else:
        coords = causet.coordinates()
        C_dense = np.array(causet.C, dtype=int)
        node_ids = list(range(causet.n))

    st = _spacetime_of(causet)
    coords = _embedding(coords, st)
    dim = coords.shape[1]
    labels = _axis_labels(st, dim)

    if dim > 3:
        warnings.warn(
            f"plot_hasse: embedding has {dim} dimensions; rendering the first 3 "
            f"axes ({labels[:3]}) and dropping {labels[3:]}.",
            UserWarning,
            stacklevel=2,
        )
        coords = coords[:, :3]
        labels = labels[:3]
        dim = 3

    # Transitive reduction (links): L = C & ~(C @ C)
    paths_len_2 = (C_dense @ C_dense) > 0
    L = C_dense & (~paths_len_2)
    link_indices = np.argwhere(L)

    edge_x, edge_y, edge_z = [], [], []

    if dim == 2:
        for i, j in link_indices:
            edge_x.extend([coords[i, 1], coords[j, 1], None])
            edge_y.extend([coords[i, 0], coords[j, 0], None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=line_width, color=line_color),
            hoverinfo="none",
            mode="lines",
        )
        node_trace = go.Scatter(
            x=coords[:, 1], y=coords[:, 0],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=coords[:, 0],
                colorscale="Viridis",
                line_width=0,
            ),
            text=[f"ID: {i}" for i in node_ids],
            hoverinfo="text",
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title or "Hasse Diagram",
            xaxis_title=labels[1],
            yaxis_title=labels[0],
            template="plotly_dark",
            showlegend=False,
        )
    else:  # dim == 3
        for i, j in link_indices:
            edge_x.extend([coords[i, 1], coords[j, 1], None])
            edge_y.extend([coords[i, 2], coords[j, 2], None])
            edge_z.extend([coords[i, 0], coords[j, 0], None])
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=line_width, color=line_color),
            hoverinfo="none",
            mode="lines",
        )
        node_trace = go.Scatter3d(
            x=coords[:, 1], y=coords[:, 2], z=coords[:, 0],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=coords[:, 0],
                colorscale="Viridis",
                line_width=0,
            ),
            text=[f"ID: {i}" for i in node_ids],
            hoverinfo="text",
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title or "Hasse Diagram (3D)",
            scene=dict(
                xaxis_title=labels[1],
                yaxis_title=labels[2],
                zaxis_title=labels[0],
            ),
            template="plotly_dark",
            showlegend=False,
        )

    return fig


def plot_causal_matrix(
    causet: CausalSet,
    max_points: int = 2000,
    force: bool = False,
    title: Optional[str] = None,
    color_scale: str = 'Greys'
):
    """Visualize the Causal Matrix (Adjacency Matrix) as a heatmap.

    Since the matrix is strictly upper triangular (for a sorted causal set),
    the heatmap will show a triangular pattern.
    """
    _check_plotly()

    indices, _ = _subset(causet.n, max_points, force, "plot_causal_matrix")

    if indices is not None:
        matrix_data = np.array(causet.C, dtype=int)[np.ix_(indices, indices)]
    else:
        matrix_data = np.array(causet.C, dtype=int)

    fig = px.imshow(
        matrix_data,
        color_continuous_scale=color_scale,
        title=title or "Causal Matrix Heatmap",
        labels=dict(x="Future Index", y="Past Index", color="Relation"),
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_side="top",
    )

    return fig
