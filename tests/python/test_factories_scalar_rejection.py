import pytest

import pycauset


def test_matrix_rejects_float_scalar_deterministically():
    with pytest.raises(TypeError) as exc:
        pycauset.matrix(1.2)
    assert "Scalars/0D are not supported" in str(exc.value)


def test_vector_rejects_float_scalar_deterministically():
    with pytest.raises(TypeError) as exc:
        pycauset.vector(1.2)
    assert "Scalars/0D are not supported" in str(exc.value)


def test_matrix_accepts_existing_native_matrix_as_noop():
    m = pycauset.FloatMatrix(2, 3)
    try:
        m2 = pycauset.matrix(m)
        assert m2 is m

        with pytest.raises(TypeError):
            pycauset.matrix(m, dtype="float32")
    finally:
        m.close()


def test_matrix_1d_returns_vector():
    v = pycauset.matrix([1, 2, 3])
    assert isinstance(v, pycauset.VectorBase)
    assert v.size() == 3
    v.close()


def test_matrix_2d_returns_matrix():
    m = pycauset.matrix([[1, 2], [3, 4]])
    assert isinstance(m, pycauset.MatrixBase)
    assert m.rows() == 2
    assert m.cols() == 2
    m.close()


def test_matrix_rejects_3d_nested_sequences_deterministically():
    with pytest.raises(TypeError) as exc:
        pycauset.matrix([[[1]]])
    assert "expects 1D or 2D" in str(exc.value)


def test_vector_accepts_existing_native_vector_as_noop():
    v = pycauset.FloatVector(5)
    try:
        v2 = pycauset.vector(v)
        assert v2 is v

        with pytest.raises(TypeError):
            pycauset.vector(v, dtype="float32")
    finally:
        v.close()
