"""PyCauset warning categories.

These exist so users can filter/suppress PyCauset warnings without
catching all UserWarning.

Keep this module lightweight and dependency-free to avoid import cycles.
"""


class PyCausetWarning(UserWarning):
    """Base warning category for all PyCauset user-facing warnings."""


class PyCausetDTypeWarning(PyCausetWarning):
    """Warnings about dtype/promotion/accumulator choices."""


class PyCausetOverflowRiskWarning(PyCausetWarning):
    """Heuristic warnings about possible overflow (preflight risk checks)."""


class PyCausetPerformanceWarning(PyCausetWarning):
    """Warnings about likely performance pitfalls (e.g., slow fallbacks)."""
