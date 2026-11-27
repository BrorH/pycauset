"""obj.properties: user notes, saved and loaded (F-092)."""
import numpy as np

import pycauset as pc


def test_notes_roundtrip(tmp_path):
    m = pc.matrix(np.eye(2))
    m.properties["source"] = "sprinkling"
    m.properties["seed"] = 4
    m.properties["params"] = {"d": 2, "rho": 1.5}
    pc.save(m, tmp_path / "m.pycauset")
    w = pc.load(tmp_path / "m.pycauset")
    assert w.properties == {"source": "sprinkling", "seed": 4, "params": {"d": 2, "rho": 1.5}}


def test_notes_are_not_structure_and_not_carried_to_views():
    m = pc.matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    m.properties["is_symmetric"] = True  # just a note: nothing reads it
    assert pc.determinant(m) == -2.0
    assert m.T.properties == {}


def test_keys_must_be_strings():
    m = pc.matrix(np.eye(2))
    try:
        m.properties = {1: "x"}
    except TypeError:
        return
    raise AssertionError("expected TypeError")
