try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None
    px = None

import random
import numpy as np
from typing import Optional, Union, List
from .causet import CausalSet

def _check_plotly():
    if go is None:
        raise ImportError("Plotly is required for visualization. Install it with 'pip install plotly'.")

def plot_embedding(
    causet: CausalSet, 
    sample_size: int = 50000, 
    title: Optional[str] = None,
    marker_size: int = 2
):
    """
    Visualize the spacetime embedding of the Causal Set.
    
    Args:
        causet: The CausalSet to plot.
        sample_size: Maximum number of points to plot. If N > sample_size, a random subset is used.
        title: Plot title.
        marker_size: Size of the scatter points.
        
    Returns:
        plotly.graph_objects.Figure: The interactive plot.
    """
    _check_plotly()
    
    n = causet.n
    indices = None
    note = ""
    
    if n > sample_size:
        # Subsampling
        # Use a fixed seed for deterministic subsampling across runs if the set is the same
        rng = random.Random(42)
        indices = sorted(rng.sample(range(n), sample_size))
        note = f" (Subsample of {sample_size} points)"
    
    # Retrieve coordinates (N x D)
    coords = causet.coordinates(indices=indices)
    
    # Coordinate Transformation
    if hasattr(causet._spacetime, 'transform_coordinates'):
        coords = causet._spacetime.transform_coordinates(coords)
        
    dim = coords.shape[1]
    
    # Boundary Visualization
    boundary_traces = []
    if hasattr(causet._spacetime, 'get_boundary'):
        boundaries = causet._spacetime.get_boundary()
        for b_coords in boundaries:
            if dim == 2:
                boundary_traces.append(go.Scatter(
                    x=b_coords[:, 1], y=b_coords[:, 0],
                    mode='lines',
                    line=dict(color='white', width=2),
                    name='Boundary',
                    hoverinfo='skip'
                ))
            elif dim == 3:
                boundary_traces.append(go.Scatter3d(
                    x=b_coords[:, 1], y=b_coords[:, 2], z=b_coords[:, 0],
                    mode='lines',
                    line=dict(color='white', width=2),
                    name='Boundary',
                    hoverinfo='skip'
                ))

    # Coordinate mapping:
    # 0 -> Time (T)
    # 1 -> X
    # 2 -> Y (if exists)
    # 3 -> Z (if exists)
    
    if dim == 2:
        # 2D Plot: X vs T
        data = [go.Scatter(
            x=coords[:, 1],
            y=coords[:, 0],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=coords[:, 0], # Color by time
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time")
            ),
            name='Events'
        )] + boundary_traces
        
        fig = go.Figure(data=data)
        fig.update_layout(
            title=title or f"2D Spacetime Embedding{note}",
            xaxis_title="X (Space)",
            yaxis_title="T (Time)",
            template="plotly_dark",
            showlegend=False
        )
        
    elif dim >= 3:
        # 3D Plot: X, Y, T
        # Note: Plotly 3D uses (x, y, z). We map (X, Y, T) -> (x, y, z)
        # So Z-axis is Time.
        
        x_data = coords[:, 1]
        y_data = coords[:, 2]
        z_data = coords[:, 0] # Time
        
        data = [go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=z_data,
                colorscale='Viridis',
                opacity=0.8,
                showscale=True,
                colorbar=dict(title="Time")
            ),
            name='Events'
        )] + boundary_traces
        
        fig = go.Figure(data=data)
        
        fig.update_layout(
            title=title or f"{dim}D Spacetime Embedding{note}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="T (Time)"
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False
        )
        
    return fig

def plot_hasse(
    causet: CausalSet,
    title: Optional[str] = None,
    marker_size: int = 5,
    line_width: int = 1,
    line_color: str = 'rgba(255, 255, 255, 0.3)'
):
    """
    Generate a Hasse diagram of the Causal Set.
    
    A Hasse diagram displays the transitive reduction of the partial order.
    Elements are placed at their spacetime coordinates, and lines are drawn
    only between immediate causal neighbors (links).
    
    Args:
        causet: The CausalSet to visualize.
        title: Plot title.
        marker_size: Size of the element markers.
        line_width: Width of the link lines.
        line_color: Color of the link lines.
        
    Returns:
        plotly.graph_objects.Figure: The interactive plot.
        
    Raises:
        ValueError: If the causal set is too large (> 500 elements) for a Hasse diagram.
    """
    _check_plotly()
    
    if causet.n > 500:
        raise ValueError(f"Causal set is too large for a Hasse diagram (N={causet.n} > 500). "
                         "Hasse diagrams are computationally expensive and visually cluttered for large sets.")

    # 1. Get Coordinates (Cartesian)
    coords = causet.coordinates()
    
    # Coordinate Transformation
    if hasattr(causet._spacetime, 'transform_coordinates'):
        coords = causet._spacetime.transform_coordinates(coords)
        
    dim = coords.shape[1]

    # 2. Compute Transitive Reduction (Links)
    # L = C & ~(C @ C)
    # We need the dense matrix for this
    C_dense = np.array(causet.C, dtype=int)
    # Boolean matrix multiplication to find paths of length 2
    # If (C @ C)[i, j] > 0, there is a path i -> k -> j
    # We want edges i -> j where NO such k exists.
    # Note: C is strictly upper triangular.
    
    # Optimization: We can use bitwise operations if we had bitsets, but numpy int is fine for N=500.
    # C @ C counts paths of length 2.
    paths_len_2 = (C_dense @ C_dense) > 0
    
    # The link matrix has 1 where C has 1 AND paths_len_2 has 0
    L = C_dense & (~paths_len_2)
    
    # 3. Create Plot Traces
    # We need to draw lines for every 1 in L
    link_indices = np.argwhere(L)
    
    edge_x = []
    edge_y = []
    edge_z = [] # For 3D
    
    if dim == 2:
        # 2D Lines
        for i, j in link_indices:
            edge_x.extend([coords[i, 1], coords[j, 1], None])
            edge_y.extend([coords[i, 0], coords[j, 0], None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=line_width, color=line_color),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=coords[:, 1], y=coords[:, 0],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=coords[:, 0],
                colorscale='Viridis',
                line_width=0
            ),
            text=[f"ID: {i}" for i in range(causet.n)],
            hoverinfo='text'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title or "Hasse Diagram",
            xaxis_title="X",
            yaxis_title="T",
            template="plotly_dark",
            showlegend=False
        )
        
    else:
        # 3D Lines (or Cylinder mapped to 3D)
        # Coords: 0->T(Z), 1->X, 2->Y
        # Plotly 3D: x, y, z
        # We map T -> Z
        
        for i, j in link_indices:
            edge_x.extend([coords[i, 1], coords[j, 1], None])
            edge_y.extend([coords[i, 2], coords[j, 2], None])
            edge_z.extend([coords[i, 0], coords[j, 0], None])
            
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=line_width, color=line_color),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter3d(
            x=coords[:, 1], y=coords[:, 2], z=coords[:, 0],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=coords[:, 0],
                colorscale='Viridis',
                line_width=0
            ),
            text=[f"ID: {i}" for i in range(causet.n)],
            hoverinfo='text'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title or "Hasse Diagram (3D)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="T"
            ),
            template="plotly_dark",
            showlegend=False
        )
        
    return fig

def plot_causal_matrix(
    causet: CausalSet,
    title: Optional[str] = None,
    color_scale: str = 'Greys'
):
    """
    Visualize the Causal Matrix (Adjacency Matrix) as a heatmap.
    
    Since the matrix is strictly upper triangular (for a sorted causal set),
    the heatmap will show a triangular pattern.
    
    Args:
        causet: The CausalSet to visualize.
        title: Plot title.
        color_scale: The color scale to use (e.g., 'Greys', 'Viridis').
        
    Returns:
        plotly.graph_objects.Figure: The interactive heatmap.
        
    Raises:
        ValueError: If the matrix is too large (> 2000 elements) to render effectively.
    """
    _check_plotly()
    
    if causet.n > 2000:
        raise ValueError(f"Matrix is too large to plot (N={causet.n} > 2000). "
                         "Try subsampling or visualizing a smaller region.")
                         
    # Get the dense matrix
    # We cast to integer (0/1) for plotting
    matrix_data = np.array(causet.C, dtype=int)
    
    fig = px.imshow(
        matrix_data,
        color_continuous_scale=color_scale,
        title=title or "Causal Matrix Heatmap",
        labels=dict(x="Future Index", y="Past Index", color="Relation")
    )
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_side="top"
    )
    
    return fig
