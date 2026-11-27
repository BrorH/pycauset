"""The Python MinkowskiDiamond and the native one it hands off to (spacetime.py `to_native`)
must give the same causal relation on the same points (PLAN 4.7: one Minkowski path).

d > 2 is the documented placeholder (product interval [0,1]^d, F-105), not Minkowski space;
this test pins that the two implementations agree, not that the physics is right."""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_native_and_python_causal_relations_agree(dim):
    st = pc.spacetime.MinkowskiDiamond(dim)
    c = pc.CausalSet(40, spacetime=st, seed=7)
    coords = np.asarray(c.coordinates())
    want = np.triu(st.is_causal_batch(coords), 1)  # the matrix stores u < v for i < j
    got = np.asarray(c.C).astype(bool)
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)
