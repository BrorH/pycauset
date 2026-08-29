"""R2_VIZ — CausalSet.plot_* methods, lazy top-level verbs, and the subset policy."""

import sys
import unittest
import warnings
from unittest.mock import MagicMock

try:
    import plotly  # noqa: F401
except ImportError:
    sys.modules.setdefault("plotly", MagicMock())
    sys.modules.setdefault("plotly.graph_objects", MagicMock())
    sys.modules.setdefault("plotly.express", MagicMock())

import pycauset as pc
from pycauset import spacetime as sp
from pycauset._internal.warnings import PyCausetPerformanceWarning


def _axis_title(fig, *path):
    """Walk a plotly layout path to an axis, returning its title or None (mocked)."""
    node = fig.layout
    for key in path:
        node = getattr(node, key, None)
        if node is None:
            return None
    title = getattr(node, "title", None)
    if title is None:
        return None
    text = getattr(title, "text", None)
    return text if isinstance(text, str) else None


class TestCausalSetPlotMethods(unittest.TestCase):
    def setUp(self):
        self.c = pc.CausalSet(n=100, spacetime=pc.MinkowskiDiamond(2), seed=1)

    def test_plot_embedding_method(self):
        self.assertIsNotNone(self.c.plot_embedding())

    def test_plot_hasse_method(self):
        self.assertIsNotNone(self.c.plot_hasse())

    def test_plot_causal_matrix_method(self):
        self.assertIsNotNone(self.c.plot_causal_matrix())

    def test_top_level_verbs(self):
        self.assertIsNotNone(pc.plot_embedding(self.c))
        self.assertIsNotNone(pc.plot_hasse(self.c))
        self.assertIsNotNone(pc.plot_causal_matrix(self.c))

    def test_show(self):
        pc.show(self.c)  # must not raise


class TestSubsetPolicy(unittest.TestCase):
    def setUp(self):
        self.c = pc.CausalSet(n=100, spacetime=pc.MinkowskiDiamond(2), seed=1)

    def test_subsample_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.c.plot_embedding(max_points=10)
        self.assertTrue(any(issubclass(x.category, PyCausetPerformanceWarning) for x in w))

    def test_force_bypasses(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.c.plot_embedding(max_points=10, force=True)
        self.assertFalse(any(issubclass(x.category, PyCausetPerformanceWarning) for x in w))


class TestAuthoredShapes(unittest.TestCase):
    """The spacetime's to_embedding / boundary / display_axes drive the plot."""

    def test_diamond_axis_labels(self):
        c = pc.CausalSet(n=20, spacetime=pc.MinkowskiDiamond(2), seed=1)
        fig = c.plot_embedding()
        self.assertEqual(_axis_title(fig, "xaxis"), "x")
        self.assertEqual(_axis_title(fig, "yaxis"), "t")

    def test_cylinder_3d_boundary(self):
        c = pc.CausalSet(n=20, spacetime=pc.MinkowskiCylinder(2, 5.0, 3.0), seed=1)
        fig = c.plot_embedding()
        # event trace + two boundary circles (top and bottom)
        self.assertGreaterEqual(len(fig.data), 3)
        self.assertEqual(_axis_title(fig, "scene", "xaxis"), "x")
        self.assertEqual(_axis_title(fig, "scene", "yaxis"), "y")
        self.assertEqual(_axis_title(fig, "scene", "zaxis"), "t")

    def test_higher_d_warns_not_silently_truncated(self):
        c = pc.CausalSet(n=20, spacetime=pc.MinkowskiBox(4, 2.0, 1.0), seed=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = c.plot_embedding()
        self.assertIsNotNone(fig)
        self.assertTrue(any("rendering the first 3" in str(x.message) for x in w))

    def test_geometry_free_custom_spacetime_renders_raw(self):
        class Raw(sp.Spacetime):
            def dimension(self):
                return 2

            def volume(self):
                return 1.0

            def sample(self, rng, n):
                return rng.uniform(size=(n, 2))

            def is_causal(self, u, v):
                return u[0] < v[0]

        c = pc.CausalSet(n=15, spacetime=Raw(), seed=2)
        fig = c.plot_embedding()
        self.assertIsNotNone(fig)
        # no authored shape → generic fallback axis labels
        self.assertEqual(_axis_title(fig, "xaxis"), "c1")
        self.assertEqual(_axis_title(fig, "yaxis"), "c0")


if __name__ == "__main__":
    unittest.main()
