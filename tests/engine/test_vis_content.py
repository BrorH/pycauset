"""What the plots draw (review 3.12 C-1 to C-7): figures were only checked to exist."""
import random
import warnings

import numpy as np
import pytest

import pycauset as pc
from pycauset.causet import CausalSet

pytest.importorskip("plotly")


@pytest.fixture
def c():
    return pc.causet(n=300, seed=1)


def test_hasse_edges_are_the_links(c):
    fig = pc.plot_hasse(c)
    assert sum(x is None for x in fig.data[0].x) == int(np.asarray(c.links()).sum())


def test_heatmap_is_the_matrix_in_bytes(c):
    z = np.asarray(pc.plot_causal_matrix(c).data[0].z)
    assert z.dtype == np.uint8 and np.array_equal(z, np.asarray(c.C).astype(np.uint8))


def test_embedding_coordinates_and_relations(c):
    fig = pc.plot_embedding(c, show_relations=True, max_relations=10**9)
    xy = c.coordinates()
    events = [t for t in fig.data if t.name == "Events"][0]
    np.testing.assert_allclose(events.x, xy[:, 1])
    np.testing.assert_allclose(events.y, xy[:, 0])
    rel = [t for t in fig.data if t.name == "Causal relations"][0]
    assert sum(x is None for x in rel.x) == int(np.asarray(c.C).sum())


def test_matrix_built_causets_have_no_embedding(tmp_path):
    """C-1: plotted at coordinates from a random seed and the default diamond."""
    m = CausalSet(n=30, matrix=pc.causal_matrix(np.triu(np.ones((30, 30), bool), 1)))
    for plot in (pc.plot_embedding, pc.plot_hasse):
        with pytest.raises(ValueError, match="no coordinates"):
            plot(m)
    pc.plot_causal_matrix(m)  # the matrix itself plots
    p = tmp_path / "m.pycauset"
    m.save(p)
    with pytest.raises(ValueError, match="no coordinates"):
        pc.load(p).coordinates()


def test_subsets_are_selected_in_the_engine(c):
    """C-2: the whole matrix was exported (as int64) before subsampling."""
    big = pc.causet(n=3000, seed=2)
    old = pc._internal.export_guard.get_max_bytes()
    try:
        pc.set_export_max_bytes(4 * 500 * 500)  # far below the full 3000^2 export
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pc.PyCausetPerformanceWarning)
            pc.plot_causal_matrix(big, max_points=500)
            pc.plot_hasse(big, max_points=500)
            pc.plot_embedding(big, max_points=500, show_relations=True)
    finally:
        pc.set_export_max_bytes(old)


def test_subsets_titles_dimensions_warnings_and_random(c):
    """C-4 / C-6: a subsampled Hasse diagram said nothing about being a sub-order; a 4-D
    embedding was titled 3D; warnings pointed into the library; the global random stream moved."""
    with pytest.warns(pc.PyCausetPerformanceWarning, match="sub-order") as rec:
        fig = pc.plot_hasse(c, max_points=100)
    assert "sub-order" in fig.layout.title.text and all(r.filename == __file__ for r in rec)
    with pytest.warns(pc.PyCausetPerformanceWarning):
        assert "subsample" in pc.plot_causal_matrix(c, max_points=100).layout.title.text
    c4 = pc.causet(n=40, spacetime=pc.spacetime.MinkowskiDiamond(4), seed=3)
    with pytest.warns(pc.PyCausetWarning) as rec:
        assert pc.plot_embedding(c4).layout.title.text.startswith("4D")
    assert all(r.filename == __file__ for r in rec)
    random.seed(0)
    a = random.random()
    random.seed(0)
    with pytest.warns(pc.PyCausetPerformanceWarning):
        pc.plot_hasse(c, max_points=100)
    assert random.random() == a


def test_force_renders_everything_above_the_coordinate_guard(c, monkeypatch):
    """C-3: force=True raised above 100,000 elements (the guard raised UserWarning)."""
    seen = {}
    real = CausalSet.coordinates

    def coordinates(self, indices=None, force=False):
        seen["force"] = force
        return real(self, indices=indices, force=force)

    monkeypatch.setattr(CausalSet, "coordinates", coordinates)
    pc.plot_embedding(c, force=True)
    assert seen["force"] is True
    c._n = 200_000
    with pytest.raises(ValueError):
        real(c)  # the guard is a ValueError now
