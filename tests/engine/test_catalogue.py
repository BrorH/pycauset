"""The structure x dtype catalogue (review N.5; include/pycauset/core/DTypeDispatch.hpp).

Values of every operation on every type are the atlas's job (tests/atlas). This file pins what
exists: which (structure, dtype) pairs construct, that each saves and loads as itself, and
the structural rules of the symmetric family (symmetric, antisymmetric, Hermitian,
skew-Hermitian; a complex symmetric matrix is not Hermitian).
"""
from __future__ import annotations

import numpy as np
import pytest

import pycauset as pc
from tests.conftest import to_np

DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float16", "float32", "float64", "complex_float16", "complex_float32", "complex_float64"]
COMPLEX = {"complex_float16", "complex_float32", "complex_float64"}
SIGNED = set(DTYPES) - {"bool", "uint8", "uint16", "uint32", "uint64"}

CLASSES = {
    "triangular": pc.TriangularMatrix,
    "diagonal": pc.DiagonalMatrix,
    "identity": pc.IdentityMatrix,
    "symmetric": pc.SymmetricMatrix,
    "antisymmetric": pc.AntiSymmetricMatrix,
    "hermitian": pc.HermitianMatrix,
    "skew_hermitian": pc.SkewHermitianMatrix,
}


def offered(structure: str, dtype: str) -> bool:
    """The catalogue as data: change it only together with `offered` in DTypeDispatch.hpp."""
    if structure == "antisymmetric":
        return dtype in SIGNED
    if structure in ("hermitian", "skew_hermitian"):
        return dtype in COMPLEX
    return True


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("structure", sorted(CLASSES))
def test_catalogue(structure, dtype, tmp_path):
    cls = CLASSES[structure]
    if not offered(structure, dtype):
        with pytest.raises(ValueError):
            cls(4, dtype=dtype)
        return
    m = cls(4, dtype=dtype)
    assert type(m) is cls and m.dtype == dtype and m.shape == (4, 4)
    path = tmp_path / "m.pycauset"
    pc.save(m, path)
    back = pc.load(path)
    try:
        assert type(back) is cls and back.dtype == dtype
        np.testing.assert_array_equal(to_np(back), to_np(m))
    finally:
        back.close()


def _hermitian_pair():
    h = pc.HermitianMatrix(3, dtype="complex_float64")
    h[0, 0] = 2.0
    h[0, 1] = 1 + 2j
    h[2, 1] = 3 - 1j  # set through the lower triangle: stored as its conjugate
    return h, to_np(h)


def test_hermitian_reads_and_views():
    h, a = _hermitian_pair()
    np.testing.assert_array_equal(a, a.conj().T)
    np.testing.assert_array_equal(to_np(h.T), a.T)          # O(1): conj(A)
    np.testing.assert_array_equal(to_np(h.H), a.conj().T)   # A^H == A
    np.testing.assert_array_equal(to_np(h.conj()), a.conj())
    np.testing.assert_array_equal(to_np(h * 2j), a * 2j)


def test_skew_hermitian_reads_and_views():
    k = pc.SkewHermitianMatrix(3, dtype="complex_float32")
    k[0, 0] = 3j
    k[0, 2] = 2 + 1j
    b = to_np(k)
    np.testing.assert_array_equal(b, -b.conj().T)
    np.testing.assert_array_equal(to_np(k.T), b.T)
    np.testing.assert_array_equal(to_np(k.H), b.conj().T)


def test_complex_symmetric_is_not_hermitian():
    s = pc.SymmetricMatrix(2, dtype="complex_float64")
    s[0, 1] = 1 + 2j
    a = to_np(s)
    assert a[1, 0] == 1 + 2j  # A == A^T, not A^H
    np.testing.assert_array_equal(to_np(s.T), a.T)


@pytest.mark.parametrize("cls,value,message", [
    (pc.HermitianMatrix, 1j, "real"),
    (pc.SkewHermitianMatrix, 1.0, "imaginary"),
    (pc.AntiSymmetricMatrix, 1.0, "zero"),
])
def test_diagonal_rules(cls, value, message):
    m = cls(2, dtype="complex_float64")
    with pytest.raises(ValueError, match=message):
        m[1, 1] = value


def test_hermitian_needs_a_complex_dtype():
    with pytest.raises(ValueError, match="complex"):
        pc.HermitianMatrix(3, dtype="float64")
