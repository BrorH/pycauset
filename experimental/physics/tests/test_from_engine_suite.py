"""Physics tests that lived in tests/engine until physics was parked (R1)."""
from __future__ import annotations

import numpy as np

import pycauset as pc


def test_causal_set_with_an_explicit_spacetime():
    st = pc.spacetime.MinkowskiDiamond(2)
    c = pc.CausalSet(n=20, spacetime=st, seed=1)
    assert c.N == 20 and c.spacetime is st

def test_causal_set_n_property():
    assert pc.CausalSet(n=7, seed=3).n == 7

def test_causal_set_round_trip(tmp_path):
    c = pc.CausalSet(30, spacetime=pc.spacetime.MinkowskiDiamond(2), seed=4)
    path = tmp_path / "c.pycauset"
    c.save(str(path))
    back = pc.CausalSet.load(str(path))
    assert back.n == 30
    np.testing.assert_array_equal(np.asarray(back.C), np.asarray(c.C))

def test_causal_set_on_a_cylinder_round_trip(tmp_path):
    st = pc.spacetime.MinkowskiCylinder(2, height=2.0, circumference=3.0)
    c = pc.CausalSet(25, spacetime=st, seed=2)
    path = tmp_path / "cyl.pycauset"
    c.save(str(path))
    back = pc.CausalSet.load(str(path))
    assert type(back.spacetime).__name__ == "MinkowskiCylinder"
    np.testing.assert_array_equal(np.asarray(back.C), np.asarray(c.C))
