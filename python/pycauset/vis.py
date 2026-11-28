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

from ._internal.warnings import PyCausetPerformanceWarning, PyCausetWarning
from .causet import CausalSet

import os as _os

_PACKAGE_DIR = _os.path.dirname(_os.path.abspath(__file__))

# Relation lines are drawn among at most this many points: the pairs come from a dense export
# of their causal sub-matrix (n^2 bytes; 2.5 GB at 50,000 points, review 3.12 C-2).
_MAX_RELATION_POINTS = 5000


def _warn(message: str, category: type) -> None:
    """A warning attributed to the user's line, the first frame outside pycauset (the plots are
    reached through pc.plot_*, CausalSet.plot_* or vis.plot_*: a fixed stacklevel pointed into
    the library; review 3.12 C-6)."""
    import sys

    level, frame = 2, sys._getframe(1)  # stacklevel 2 is _warn's caller
    while frame is not None and _os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR + _os.sep):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)


def _engine_subset(causet: CausalSet, indices) -> np.ndarray:
    """The causal sub-matrix on `indices` (all elements when None) as a NumPy bool array. The
    selection runs in the engine, so only the sub-matrix is exported: the whole matrix was
    exported first (as int64 for the heatmap: 22 GB at n = 50,000 for a 2000^2 image, review
    3.12 C-2)."""
    C = causet.C
    if indices is not None:
        C = C[list(indices)][:, list(indices)]
    return np.asarray(C).astype(bool, copy=False)


def _subset(n: int, max_points: Optional[int], force: bool, kind: str):
    """Return (indices_or_None, note) for the large-set policy .

    Above ``max_points`` a seeded random subset is drawn and a
    `PyCausetPerformanceWarning` is emitted; ``force=True`` (or ``max_points=None``)
    renders everything.
    """
    if max_points is None or force or n <= max_points:
        return None, ""
    rng = random.Random(42)
    indices = sorted(rng.sample(range(n), max_points))
    note = f" (subsample of {max_points} of {n} points)"
    extra = ("; its lines are the links of that sub-order, not all links of the causal set"
             if kind == "plot_hasse" else "")
    _warn(
        f"{kind} of a {n}-element causal set: plotting a seeded random subset of "
        f"{max_points} elements{extra}. Pass force=True (or max_points=None) to render all.",
        PyCausetPerformanceWarning,
    )
    return indices, note


def _check_plotly():
    if go is None:
        raise ImportError("pycauset.vis needs plotly: pip install \"pycauset[plot]\" (or pip install plotly)")


def _spacetime_of(causet: CausalSet):
    """The causet's spacetime (or None if unavailable)."""
    return getattr(causet, "_spacetime", None)


def _boundary_paths(st):
    """Boundary paths in embedding coordinates (default: none).

    Uses the spacetime's ``boundary()``, in its own coordinates.
    """
    if st is None:
        return []
    fn = getattr(st, "boundary", None)
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
        if b.ndim != 2 or b.shape[1] < dim:  # a spacetime's bug: it was dropped silently (C-6)
            raise ValueError(f"spacetime boundary path of shape {b.shape} does not fit a {dim}-D embedding")
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
    force: bool = False,
    title: Optional[str] = None,
    marker_size: int = 2,
    show_relations: bool = False,
    max_relations: int = 20000
):
    """Visualize the spacetime embedding of the Causal Set.

    The spacetime's ``boundary()`` / ``display_axes()`` draw its outline and name the
    axes (generic ``c0, c1, ...`` labels without them). Embedding dimensions beyond 3 are shown
    as the first three axes with an explicit warning (never silently truncated).

    When ``show_relations`` is true, a faint line is drawn for every causal pair
    ``A < B`` among the plotted points (the full relation, not just the Hasse
    links). This materialises the dense causal matrix, so it is intended for
    small/medium causets; if there are more than ``max_relations`` pairs, a seeded
    random subset of them is drawn with a warning.
    """
    _check_plotly()

    n = causet.n
    indices, note = _subset(n, max_points, force, "plot_embedding")

    st = _spacetime_of(causet)
    # (force: "render everything" raised above 100,000 elements, review 3.12 C-3.)
    coords = np.asarray(causet.coordinates(indices=indices, force=indices is None), dtype=float)
    dim = full_dim = coords.shape[1]
    labels = _axis_labels(st, dim)

    if dim > 3:
        _warn(
            f"plot_embedding: embedding has {dim} dimensions; rendering the first 3 "
            f"axes ({labels[:3]}) and dropping {labels[3:]}.",
            PyCausetWarning,
        )
        coords = coords[:, :3]
        labels = labels[:3]
        dim = 3

    boundary_traces = _boundary_traces(_boundary_paths(st), dim)

    # Optional causal-relation edges: a faint line for every pair A < B among the
    # plotted points (the full relation, not the Hasse links).
    edge_traces = []
    if show_relations:
        points = list(range(len(coords)))  # positions in `coords`
        if len(points) > _MAX_RELATION_POINTS:
            points = sorted(random.Random(7).sample(points, _MAX_RELATION_POINTS))
            _warn(
                f"plot_embedding: relations are drawn among a seeded random {_MAX_RELATION_POINTS} "
                f"of the {len(coords)} plotted points (their causal sub-matrix is exported).",
                PyCausetPerformanceWarning,
            )
        elements = points if indices is None else [indices[p] for p in points]
        # Strictly upper triangular already (time-labelled, and the selection keeps the order).
        src, dst = np.nonzero(_engine_subset(causet, elements))
        src, dst = np.asarray(points)[src], np.asarray(points)[dst]
        if len(src) > max_relations:
            _warn(
                f"plot_embedding: {len(src)} causal pairs exceed "
                f"max_relations={max_relations}; drawing a seeded random subset.",
                PyCausetPerformanceWarning,
            )
            rng = np.random.default_rng(42)
            sel = rng.choice(len(src), size=max_relations, replace=False)
            src, dst = src[sel], dst[sel]

        edge_x, edge_y, edge_z = [], [], []
        for i, j in zip(src.tolist(), dst.tolist()):
            if dim == 2:
                edge_x.extend([coords[i, 1], coords[j, 1], None])
                edge_y.extend([coords[i, 0], coords[j, 0], None])
            else:
                edge_x.extend([coords[i, 1], coords[j, 1], None])
                edge_y.extend([coords[i, 2], coords[j, 2], None])
                edge_z.extend([coords[i, 0], coords[j, 0], None])

        if dim == 2:
            edge_traces = [go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=1, color="rgba(255, 255, 255, 0.15)"),
                hoverinfo="none", name="Causal relations",
            )]
        else:
            edge_traces = [go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z, mode="lines",
                line=dict(width=1, color="rgba(255, 255, 255, 0.15)"),
                hoverinfo="none", name="Causal relations",
            )]

    # Embedding contract: column 0 is time (vertical), columns 1.. are spatial.
    if dim == 2:
        data = edge_traces + [go.Scatter(
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
        data = edge_traces + [go.Scatter3d(
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
            title=title or f"{full_dim}D Spacetime Embedding{note}",
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
    between immediate causal neighbors (links, from the engine's transitive reduction).
    Dimensions beyond 3 are shown as the first three axes with a warning.
    """
    _check_plotly()

    indices, note = _subset(causet.n, max_points, force, "plot_hasse")

    if indices is not None:  # the Hasse diagram of the induced sub-order (said so: C-4)
        from . import causal_matrix

        coords = causet.coordinates(indices=indices)
        sub = _engine_subset(causet, indices)
        L = CausalSet(n=len(indices), matrix=causal_matrix(sub), validate=False).links()
        node_ids = indices
        note = f" of a {len(indices)}-element sub-order{note}"
    else:
        coords = causet.coordinates(force=True)
        L = causet.links()  # the engine's transitive reduction
        node_ids = list(range(causet.n))

    st = _spacetime_of(causet)
    coords = np.asarray(coords, dtype=float)
    dim = coords.shape[1]
    labels = _axis_labels(st, dim)

    if dim > 3:
        _warn(
            f"plot_hasse: embedding has {dim} dimensions; rendering the first 3 "
            f"axes ({labels[:3]}) and dropping {labels[3:]}.",
            PyCausetWarning,
        )
        coords = coords[:, :3]
        labels = labels[:3]
        dim = 3

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
            title=title or f"Hasse Diagram{note}",
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
            title=title or f"Hasse Diagram{note}",
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
    color_scale: Optional[str] = None
):
    """Visualize the causal matrix as a heatmap.

    The causal matrix is boolean: a bright cell means element ``i`` is in the
    causal past of element ``j``, a dark cell means it is not. For a causal set
    whose elements are labelled by time the matrix is strictly upper triangular,
    so the plot shows a crisp triangle.

    ``color_scale`` accepts any Plotly continuous colorscale name (``"Greys"``,
    ``"Viridis"``, ...). The default is a two-tone scale (dark ``0``, teal ``1``)
    tuned for the dark template; it reads as a boolean image rather than a
    gradient.
    """
    _check_plotly()

    indices, note = _subset(causet.n, max_points, force, "plot_causal_matrix")
    matrix_data = _engine_subset(causet, indices).view(np.uint8)  # 1 byte per cell (it was int64)

    if color_scale is None:
        color_scale = [[0.0, "#10131a"], [1.0, "#14b8a6"]]

    fig = px.imshow(
        matrix_data,
        color_continuous_scale=color_scale,
        title=title or f"Causal Matrix Heatmap{note}",
        labels=dict(x="Future index", y="Past index"),
        aspect="equal",
        zmin=0,
        zmax=1,
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_side="top",
        coloraxis_colorbar=dict(
            title="Related",
            tickvals=[0, 1],
            ticktext=["no", "yes"],
            thickness=14,
        ),
    )

    return fig
