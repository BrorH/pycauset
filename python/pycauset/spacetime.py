"""Spacetimes to sprinkle causal sets into.

R1 offers the Minkowski causal diamond in any dimension; the rest of the spacetime library
(cylinders, boxes, de Sitter, FLRW, Schwarzschild, custom spacetimes) is parked in
`experimental/physics`. Sampling and causality are one native implementation
(`include/pycauset/causet/Spacetime.hpp`); this class only describes it.
"""
from __future__ import annotations

import math
from importlib import import_module
from typing import Sequence

import numpy as np

_native = import_module("._pycauset", package=__package__)


class MinkowskiDiamond:
    """The causal diamond between (t, x) = (0, 0) and (1, 0) in d-dimensional Minkowski space
    (c = 1): the points with |x| < min(t, 1 - t). Coordinates are (t, x_1, ..., x_{d-1}).

    Sprinkling is uniform in the diamond; `u` precedes `v` when
    (t_v - t_u)^2 > |x_v - x_u|^2 and t_v > t_u.
    """

    def __init__(self, dimension: int = 2):
        self._native = _native._MinkowskiDiamond(int(dimension))

    def dimension(self) -> int:
        return self._native.dimension()

    def volume(self) -> float:
        """Spacetime volume of the diamond (1/2 in d = 2)."""
        return self._native.volume()

    def is_causal(self, u: Sequence[float], v: Sequence[float]) -> bool:
        """True when point `u` is in the causal past of point `v`."""
        return bool(self._native.causality(list(map(float, u)), list(map(float, v))))

    # Plot hooks (pycauset.vis): the outline of the region and the axis names. Coordinates
    # are already (t, x, ...), so no embedding transform.
    def boundary(self):
        """Outline paths, columns (t, x[, y]): the diamond in 2D; in 3D its rim circle and
        lines from the rim to both tips. None drawn above 3D."""
        d = self.dimension()
        if d == 2:
            return [np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.5, -0.5], [0.0, 0.0]])]
        if d == 3:
            a = np.linspace(0.0, 2 * math.pi, 65)
            rim = np.column_stack((np.full_like(a, 0.5), 0.5 * np.cos(a), 0.5 * np.sin(a)))
            spokes = []
            for k in range(4):
                x, y = 0.5 * math.cos(k * math.pi / 2), 0.5 * math.sin(k * math.pi / 2)
                spokes.append(np.array([[0.0, 0.0, 0.0], [0.5, x, y], [1.0, 0.0, 0.0]]))
            return [rim] + spokes
        return []

    def display_axes(self):
        return ["t", "x", "y", "z"][: self.dimension()] if self.dimension() <= 4 else None

    def _native_spacetime(self):
        return self._native

    def __reduce__(self):
        # Pickling (a CausalSet holds its spacetime): rebuilt from the dimension.
        return (type(self), (self.dimension(),))

    def __repr__(self) -> str:
        return f"MinkowskiDiamond(dimension={self.dimension()})"
