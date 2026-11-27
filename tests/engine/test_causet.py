"""Causal sets in R1: sprinkling into the d-dimensional Minkowski diamond, the order methods,
synthetic orders, validation and persistence. Geometry is checked against an independent
NumPy statement of the Minkowski interval; orders against their definitions."""
from __future__ import annotations

import math

import numpy as np
import pytest

import pycauset as pc


def _minkowski_order(x: np.ndarray) -> np.ndarray:
    """u < v iff t_v > t_u and (t_v - t_u)^2 > |x_v - x_u|^2 (NumPy reference)."""
    dt = x[None, :, 0] - x[:, None, 0]
    dx2 = ((x[None, :, 1:] - x[:, None, 1:]) ** 2).sum(-1)
    return (dt > 0) & (dt * dt > dx2)


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_sprinkle_is_minkowski_in_every_dimension(d):
    """F-105: d > 2 used a coordinate-wise order on a cube (not Minkowski space)."""
    c = pc.causet(n=300, spacetime=pc.spacetime.MinkowskiDiamond(d), seed=11)
    x = c.coordinates()
    assert x.shape == (300, d)
    r = np.linalg.norm(x[:, 1:], axis=1)
    assert np.all(r < np.minimum(x[:, 0], 1 - x[:, 0]))  # inside the diamond
    assert np.all(np.diff(x[:, 0]) >= 0)  # time-labelled
    np.testing.assert_array_equal(np.asarray(c.C).astype(bool), np.triu(_minkowski_order(x), 1))


def test_sprinkle_is_seeded_and_a_valid_order():
    a = pc.causet(n=200, spacetime=pc.spacetime.MinkowskiDiamond(3), seed=5)
    b = pc.causet(n=200, spacetime=pc.spacetime.MinkowskiDiamond(3), seed=5)
    np.testing.assert_array_equal(np.asarray(a.C), np.asarray(b.C))
    assert type(a.C).__name__ == "TriangularMatrix" and a.C.dtype == "bool"
    a.validate()  # strict partial order


@pytest.mark.parametrize("d", [2, 3, 4])
def test_diamond_volume(d):
    ball = math.pi ** ((d - 1) / 2) / math.gamma((d - 1) / 2 + 1)
    assert pc.spacetime.MinkowskiDiamond(d).volume() == pytest.approx(2 * ball * 0.5**d / d)
    assert pc.spacetime.MinkowskiDiamond(2).volume() == pytest.approx(0.5)


def test_density_sprinkle_draws_a_poisson_count():
    st = pc.spacetime.MinkowskiDiamond(2)
    ns = [pc.causet(density=1000, spacetime=st, seed=s).n for s in range(20)]
    assert abs(np.mean(ns) - 500) < 30  # E[N] = rho * volume = 500


@pytest.mark.parametrize("d", [2, 3, 4])
def test_myrheim_meyer_recovers_the_dimension(d):
    c = pc.causet(n=2000, spacetime=pc.spacetime.MinkowskiDiamond(d), seed=3)
    assert abs(c.myrheim_meyer_dimension() - d) < 0.25


def test_order_methods_on_a_chain_and_an_antichain():
    chain = pc.synthetic.chain(6)
    assert list(chain.longest_chain()) == list(range(6))
    assert chain.is_chain(range(6)) and not chain.is_antichain([0, 1])
    assert int(chain.links().sum()) == 5
    assert list(chain.past(3)) == [0, 1, 2] and list(chain.future(3)) == [4, 5]
    assert list(chain.interval(1, 4)) == [2, 3]
    assert [list(layer) for layer in chain.layers()] == [[k] for k in range(6)]
    anti = pc.synthetic.antichain(5)
    assert len(anti.longest_chain()) == 1 and anti.is_antichain(range(5))
    assert anti.relation_fraction() == 0.0 and chain.relation_fraction() == 1.0


def test_links_and_layers_of_a_sprinkle_match_the_definitions():
    c = pc.causet(n=150, seed=8)
    C = np.asarray(c.C).astype(bool)
    Ci = C.astype(np.int64)
    np.testing.assert_array_equal(c.links(), C & ~((Ci @ Ci) > 0))
    height = len(c.longest_chain())
    assert height == len(c.layers())
    assert c.is_chain(c.longest_chain())


def test_synthetic_orders_are_valid():
    for c in (pc.synthetic.transitive_percolation(0.1, 60, seed=1), pc.synthetic.product_order([3, 4])):
        c.validate()


def test_validation_rejects_bad_orders():
    with pytest.raises(ValueError, match="reflexive-free"):
        pc.causet(n=2, matrix=pc.matrix(np.eye(2, dtype=bool)))
    with pytest.raises(ValueError, match="transitive"):
        bad = np.zeros((3, 3), dtype=bool)
        bad[0, 1] = bad[1, 2] = True  # 0 < 1 < 2 without 0 < 2
        pc.causet(n=3, matrix=pc.causal_matrix(bad))


def test_F107_validation_counts_paths_exactly():
    """256 two-step paths wrapped to 0 in the old uint8 product: an intransitive order passed."""
    n = 258
    bad = np.zeros((n, n), dtype=bool)
    bad[0, 1:257] = True
    bad[1:257, 257] = True  # 256 paths 0 -> k -> 257, but no 0 < 257
    with pytest.raises(ValueError, match="transitive"):
        pc.causet(n=n, matrix=pc.causal_matrix(bad))


def test_causet_save_load_round_trip(tmp_path):
    c = pc.causet(n=120, spacetime=pc.spacetime.MinkowskiDiamond(4), seed=9)
    path = tmp_path / "c.pycauset"
    c.save(str(path))
    back = pc.CausalSet.load(str(path))
    assert back.n == 120 and back.spacetime.dimension() == 4
    np.testing.assert_array_equal(np.asarray(back.C), np.asarray(c.C))
    np.testing.assert_allclose(back.coordinates(), c.coordinates())
