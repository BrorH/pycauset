import numpy as np
import pycauset
import pytest


def _maybe_skip_matrix(cls_name: str):
    cls = getattr(pycauset, cls_name, None)
    if cls is None:
        pytest.skip(f"{cls_name} not available in native extension")
    return cls


def test_convert_pycauset_to_npy_and_back_matrix(tmp_path):
    _maybe_skip_matrix("FloatMatrix")
    arr = np.arange(9, dtype=np.float32).reshape(3, 3)
    m = pycauset.matrix(arr)
    src = tmp_path / "a.pycauset"
    npy = tmp_path / "a.npy"
    back = tmp_path / "b.pycauset"

    try:
        pycauset.save(m, src)
        pycauset.convert_file(src, npy)
        pycauset.convert_file(npy, back)

        loaded = pycauset.load(back)
        try:
            np.testing.assert_array_equal(np.array(loaded), arr)
        finally:
            loaded.close()
    finally:
        m.close()


def test_convert_npz_to_pycauset_vector(tmp_path):
    _maybe_skip_matrix("FloatVector")
    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    npz = tmp_path / "v.npz"
    dst = tmp_path / "v.pycauset"

    np.savez(npz, arr=vec)
    pycauset.convert_file(npz, dst, npz_key="arr")

    loaded = pycauset.load(dst)
    try:
        np.testing.assert_array_equal(np.array(loaded), vec)
    finally:
        loaded.close()


def test_load_npy_and_save_npz_helpers(tmp_path):
    _maybe_skip_matrix("FloatMatrix")
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    npy = tmp_path / "m.npy"
    np.save(npy, arr)

    m = pycauset.load_npy(npy)
    dst_npz = tmp_path / "m.npz"
    try:
        pycauset.save_npz(m, dst_npz)
        data = np.load(dst_npz)
        assert "array" in data
        np.testing.assert_array_equal(data["array"], arr)
    finally:
        m.close()
