import numpy as np
import pycauset


def test_vector_from_numpy_complex64() -> None:
    if getattr(pycauset, "ComplexFloat32Vector", None) is None:
        return

    arr = np.array([1 + 2j, 3 - 4j, 0.25 + 0.5j], dtype=np.complex64)
    v = pycauset.vector(arr)

    assert type(v).__name__ == "ComplexFloat32Vector"
    out = np.array(v)
    assert out.dtype == np.complex64
    assert np.allclose(out, arr)


def test_vector_from_numpy_complex128() -> None:
    if getattr(pycauset, "ComplexFloat64Vector", None) is None:
        return

    arr = np.array([1 + 2j, 3 - 4j, 0.25 + 0.5j], dtype=np.complex128)
    v = pycauset.vector(arr)

    assert type(v).__name__ == "ComplexFloat64Vector"
    out = np.array(v)
    assert out.dtype == np.complex128
    assert np.allclose(out, arr)
