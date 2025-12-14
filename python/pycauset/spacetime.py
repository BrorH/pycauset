try:
    from ._pycauset import MinkowskiDiamond, MinkowskiCylinder, MinkowskiBox
except ImportError:
    # Fallback for when bindings are incomplete (e.g. during optimization work)
    class MinkowskiDiamond: pass
    class MinkowskiCylinder: pass
    class MinkowskiBox: pass

import numpy as np
import math

__all__ = ["MinkowskiDiamond", "MinkowskiCylinder", "MinkowskiBox"]

# --- Common Extensions ---

# (No common extensions currently)

# --- MinkowskiDiamond Extensions ---

def _diamond_transform(self, coords):
    """
    Transform coordinates for visualization.
    2D: Rotates lightcone (u, v) to Cartesian (t, x).
    """
    if self.dimension() == 2:
        u = coords[:, 0]
        v = coords[:, 1]
        t = (u + v) / np.sqrt(2)
        x = (v - u) / np.sqrt(2)
        return np.column_stack((t, x))
    return coords

def _diamond_boundary(self):
    """
    Get the boundary of the spacetime region in transformed coordinates.
    Returns a list of arrays, where each array is a connected path.
    """
    if self.dimension() == 2:
        # Corners in (u, v): (0,0), (1,0), (1,1), (0,1), (0,0)
        corners = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ])
        return [self.transform_coordinates(corners)]
    return []

MinkowskiDiamond.transform_coordinates = _diamond_transform
MinkowskiDiamond.get_boundary = _diamond_boundary


# --- MinkowskiCylinder Extensions ---

def _cylinder_transform(self, coords):
    """
    Transform coordinates for visualization.
    2D: Maps (t, x) to 3D cylinder (z, x, y).
    """
    if self.dimension() == 2:
        t = coords[:, 0]
        x_flat = coords[:, 1]
        
        C = self.circumference
        R = C / (2 * np.pi)
        
        theta = (x_flat / C) * 2 * np.pi
        x_3d = R * np.cos(theta)
        y_3d = R * np.sin(theta)
        z_3d = t
        
        # Return as (T, X, Y) which maps to (Z, X, Y) in Plotly usually, 
        # but let's stick to a convention. 
        # vis.py used: np.column_stack((z_3d, x_3d, y_3d))
        # Let's return (z, x, y)
        return np.column_stack((z_3d, x_3d, y_3d))
    return coords

def _cylinder_boundary(self):
    """
    Get the boundary rings of the cylinder.
    """
    if self.dimension() == 2:
        C = self.circumference
        H = self.height
        
        # Generate points for the rings
        theta = np.linspace(0, 2 * np.pi, 100)
        x_flat = (theta / (2 * np.pi)) * C
        
        # Bottom ring (t=0)
        bottom = np.column_stack((np.zeros_like(x_flat), x_flat))
        
        # Top ring (t=H)
        top = np.column_stack((np.full_like(x_flat, H), x_flat))
        
        return [
            self.transform_coordinates(bottom),
            self.transform_coordinates(top)
        ]
    return []

MinkowskiCylinder.transform_coordinates = _cylinder_transform
MinkowskiCylinder.get_boundary = _cylinder_boundary


# --- MinkowskiBox Extensions ---

def _box_transform(self, coords):
    # Box coordinates are already Cartesian (t, x, y...)
    # No transform needed usually, but we can ensure consistent output
    return coords

def _box_boundary(self):
    if self.dimension() == 2:
        T = self.time_extent
        L = self.space_extent
        # Rectangle: (0,0) -> (T,0) -> (T,L) -> (0,L) -> (0,0)
        # Note: coords are (t, x)
        corners = np.array([
            [0.0, 0.0],
            [T, 0.0],
            [T, L],
            [0.0, L],
            [0.0, 0.0]
        ])
        return [corners]
    return []

MinkowskiBox.transform_coordinates = _box_transform
MinkowskiBox.get_boundary = _box_boundary



