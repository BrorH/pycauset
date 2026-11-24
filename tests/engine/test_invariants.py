"""View-metadata invariants (review/TESTS.md §3.2): every way of reading a matrix agrees.

For a catalog of objects (all dtypes, transposed and scaled views, causal matrices and
their powers, structured types, disk-backed copies) we check that element access,
NumPy export, sum, transpose, matmul with an identity and save/load round-trips all
describe the same matrix. This is the bug class of F-001 (one reader honouring view
metadata that another ignores).
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float16", "float32", "float64", "complex64", "complex128"]


def _dense(dtype: str, shape=(4, 5), seed=7):
    r = np.random.default_rng(seed).integers(0, 4, size=shape)
    return (r % 2).astype(bool) if dtype == "bool" else r.astype(dtype)


def _catalog():
    items = {}
    for dt in DTYPES:
        items[f"dense-{dt}"] = lambda dt=dt: pc.matrix(_dense(dt))
        items[f"dense-{dt}.T"] = lambda dt=dt: pc.matrix(_dense(dt)).T
    items["float64*2.5 (scalar view)"] = lambda: pc.matrix(_dense("float64")) * 2.5
    items["complex128*1j"] = lambda: pc.matrix(_dense("complex128")) * 1j
    items["int32*3 (exact)"] = lambda: pc.matrix(_dense("int32")) * 3
    items["causal"] = lambda: pc.causal_matrix(12, populate=True)
    items["causal.T"] = lambda: pc.causal_matrix(12, populate=True).T
    items["causal@causal"] = lambda: (lambda c: c @ c)(pc.causal_matrix(12, populate=True))
    items["causal*3"] = lambda: pc.causal_matrix(12, populate=True) * 3
    items["identity"] = lambda: pc.identity(5)
    items["identity*4"] = lambda: pc.identity(5) * 4
    items["diagonal"] = lambda: pc.diagonal(np.array([1.0, -2.0, 3.0]))
    sym = np.array([[1.0, 2, 3], [2, 5, 6], [3, 6, 9]])
    items["symmetric"] = lambda: pc.symmetric(sym)
    items["antisymmetric"] = lambda: pc.antisymmetric(sym - sym.T)
    return items


CATALOG = _catalog()
IDS = list(CATALOG)



def _tol(arr):
    return 2e-3 if arr.dtype in (np.float16,) else 1e-6


@pytest.fixture(params=[False, True], ids=["ram", "disk"])
def storage(request):
    if request.param:
        previous = pc.get_memory_threshold()
        pc.set_memory_threshold(1024)
        yield "disk"
        pc.set_memory_threshold(previous if previous is not None else 8 * 1024**3)
    else:
        yield "ram"


@pytest.mark.parametrize("name", IDS)
def test_element_access_matches_export(name, storage):
    m = CATALOG[name]()
    ref = to_np(m)
    rows, cols = ref.shape
    for i in range(rows):
        for j in range(cols):
            assert complex(m[i, j]) == pytest.approx(complex(ref[i, j]), rel=_tol(ref), abs=_tol(ref)), (i, j)


@pytest.mark.parametrize("name", IDS)
def test_sum_matches_export(name, storage):
    m = CATALOG[name]()
    ref = to_np(m)
    total = complex(pc.sum(m))
    expected = complex(ref.astype(np.complex128).sum())
    assert total == pytest.approx(expected, rel=max(_tol(ref), 1e-9), abs=1e-6)


@pytest.mark.parametrize("name", IDS)
def test_transpose_matches_export(name, storage):
    m = CATALOG[name]()
    np.testing.assert_array_equal(to_np(m.T), to_np(m).T)


@pytest.mark.parametrize("name", IDS)
def test_matmul_with_identity_preserves_values(name, storage):
    m = CATALOG[name]()
    ref = to_np(m)
    eye = pc.matrix(np.eye(ref.shape[1]))
    got = to_np(m @ eye)
    np.testing.assert_allclose(got.astype(np.complex128), ref.astype(np.complex128), rtol=_tol(ref), atol=_tol(ref))


@pytest.mark.parametrize("name", IDS)
def test_save_load_roundtrip(name, storage, tmp_path):
    m = CATALOG[name]()
    path = tmp_path / "m.pycauset"
    pc.save(m, path)
    loaded = pc.load(path)
    np.testing.assert_array_equal(to_np(loaded), to_np(m))
    assert type(loaded).__name__ == type(m).__name__
