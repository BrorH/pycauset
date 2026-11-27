"""Spacetimes to sprinkle causal sets into.

R1 offers the Minkowski causal diamond in any dimension; the rest of the spacetime library
(cylinders, boxes, de Sitter, FLRW, Schwarzschild, custom spacetimes) is parked in
`experimental/physics`. Sampling and causality are one native implementation
(`include/pycauset/causet/Spacetime.hpp`); this class only describes it.
"""
from __future__ import annotations

from importlib import import_module
from typing import Sequence

_native = import_module("._pycauset", package=__package__)


class MinkowskiDiamond:
    """The causal diamond between (t, x) = (0, 0) and (1, 0) in d-dimensional Minkowski space
    (c = 1): the points with |x| < min(t, 1 - t). Coordinates are (t, x_1, ..., x_{d-1}).

    Sprinkling is uniform in the diamond; `u` precedes `v` when
    (t_v - t_u)^2 > |x_v - x_u|^2 and t_v > t_u.
    """

    def __init__(self, dimension: int = 2):
        self._native = _native.MinkowskiDiamond(int(dimension))

    def dimension(self) -> int:
        return self._native.dimension()

    def volume(self) -> float:
        """Spacetime volume of the diamond (1/2 in d = 2)."""
        return self._native.volume()

    def is_causal(self, u: Sequence[float], v: Sequence[float]) -> bool:
        """True when point `u` is in the causal past of point `v`."""
        return bool(self._native.causality(list(map(float, u)), list(map(float, v))))

    def _native_spacetime(self):
        return self._native

    def __repr__(self) -> str:
        return f"MinkowskiDiamond(dimension={self.dimension()})"
