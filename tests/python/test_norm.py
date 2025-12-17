import pycauset


def test_norm_vector_l2():
    v = pycauset.vector([3.0, 4.0], dtype="float64")
    assert pycauset.norm(v) == 5.0


def test_norm_vector_scaling_respects_scalar():
    v = pycauset.vector([3.0, 4.0], dtype="float64")
    v2 = v * 2.0
    assert pycauset.norm(v2) == 10.0


def test_norm_matrix_frobenius():
    m = pycauset.matrix([[3.0, 4.0], [0.0, 0.0]], dtype="float64")
    assert pycauset.norm(m) == 5.0
