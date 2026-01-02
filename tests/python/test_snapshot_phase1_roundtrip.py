import os
from pathlib import Path

import numpy as np
import pycauset
import pytest

from pycauset._internal import persistence as _persistence


MATRIX_CASES = [
    ("FloatMatrix", np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)),
    ("Float32Matrix", np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)),
    ("Float16Matrix", np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)),
    ("IntegerMatrix", np.array([[1, 2], [3, 4]], dtype=np.int32)),
    ("Int8Matrix", np.array([[1, -2], [3, -4]], dtype=np.int8)),
    ("Int16Matrix", np.array([[1, -2], [3, -4]], dtype=np.int16)),
    ("Int64Matrix", np.array([[1, -2], [3, -4]], dtype=np.int64)),
    ("UInt8Matrix", np.array([[1, 2], [3, 4]], dtype=np.uint8)),
    ("UInt16Matrix", np.array([[1, 2], [3, 4]], dtype=np.uint16)),
    ("UInt32Matrix", np.array([[1, 2], [3, 4]], dtype=np.uint32)),
    ("UInt64Matrix", np.array([[1, 2], [3, 4]], dtype=np.uint64)),
    ("ComplexFloat32Matrix", np.array([[1 + 2j, 3 - 4j], [5 + 0j, -6 + 1j]], dtype=np.complex64)),
    ("ComplexFloat64Matrix", np.array([[1 + 2j, 3 - 4j], [5 + 0j, -6 + 1j]], dtype=np.complex128)),
    ("DenseBitMatrix", np.array([[True, False], [False, True]], dtype=bool)),
]


VECTOR_CASES = [
    ("FloatVector", np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)),
    ("Float32Vector", np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)),
    ("Float16Vector", np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float16)),
    ("IntegerVector", np.array([1, 2, 3, 4], dtype=np.int32)),
    ("Int8Vector", np.array([1, -2, 3, -4], dtype=np.int8)),
    ("Int16Vector", np.array([1, -2, 3, -4], dtype=np.int16)),
    ("Int64Vector", np.array([1, -2, 3, -4], dtype=np.int64)),
    ("UInt8Vector", np.array([1, 2, 3, 4], dtype=np.uint8)),
    ("UInt16Vector", np.array([1, 2, 3, 4], dtype=np.uint16)),
    ("UInt32Vector", np.array([1, 2, 3, 4], dtype=np.uint32)),
    ("UInt64Vector", np.array([1, 2, 3, 4], dtype=np.uint64)),
    ("ComplexFloat32Vector", np.array([1 + 2j, 3 - 4j, 5 + 0j, -6 + 1j], dtype=np.complex64)),
    ("ComplexFloat64Vector", np.array([1 + 2j, 3 - 4j, 5 + 0j, -6 + 1j], dtype=np.complex128)),
    ("BitVector", np.array([True, False, True, False], dtype=bool)),
]


@pytest.mark.parametrize("class_name,data", MATRIX_CASES)
def test_matrix_roundtrip_all_public_dtypes(tmp_path: Path, class_name: str, data: np.ndarray) -> None:
    cls = getattr(pycauset, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} not available in native extension")

    rows, cols = data.shape
    m = cls(rows, cols)
    for i in range(rows):
        for j in range(cols):
            m[i, j] = data[i, j]

    path = tmp_path / f"{class_name}.pycauset"
    try:
        pycauset.save(m, path)
        loaded = pycauset.load(path)
        try:
            assert isinstance(loaded, cls)
            np.testing.assert_array_equal(np.array(loaded), data)
        finally:
            loaded.close()
    finally:
        m.close()


@pytest.mark.parametrize("class_name,data", VECTOR_CASES)
def test_vector_roundtrip_all_public_dtypes(tmp_path: Path, class_name: str, data: np.ndarray) -> None:
    cls = getattr(pycauset, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} not available in native extension")

    n = data.shape[0]
    v = cls(n)
    for i in range(n):
        v[i] = data[i]

    path = tmp_path / f"{class_name}.pycauset"
    try:
        pycauset.save(v, path)
        loaded = pycauset.load(path)
        try:
            assert isinstance(loaded, cls)
            np.testing.assert_array_equal(np.array(loaded), data)
        finally:
            loaded.close()
    finally:
        v.close()


def test_metadata_only_update_does_not_rewrite_payload(tmp_path: Path) -> None:
    m = pycauset.FloatMatrix(3)
    for i in range(3):
        for j in range(3):
            m[i, j] = float(i * 3 + j + 1)

    path = tmp_path / "meta_guard.pycauset"
    try:
        pycauset.save(m, path)

        active_before, _, meta_before = _persistence._read_active_slot_and_typed_metadata(path)
        with path.open("rb") as f:
            f.seek(int(active_before["payload_offset"]))
            payload_before = f.read(int(active_before["payload_length"]))

        m.properties["tag"] = "roundtrip"
        pycauset.save(m, path)

        active_after, _, meta_after = _persistence._read_active_slot_and_typed_metadata(path)
        with path.open("rb") as f:
            f.seek(int(active_after["payload_offset"]))
            payload_after = f.read(int(active_after["payload_length"]))

        assert payload_before == payload_after
        assert meta_after.get("properties", {}).get("tag") == "roundtrip"
    finally:
        m.close()
        if path.exists():
            try:
                os.remove(path)
            except OSError:
                pass


def test_properties_and_cached_values_roundtrip(tmp_path: Path) -> None:
    m = pycauset.FloatMatrix(2)
    m[0, 0] = 1.0
    m[1, 1] = 2.0

    # Gospel properties and cached-derived entries.
    m.properties["label"] = "phase1"
    m.properties["trace"] = 3.0
    m.properties["determinant"] = 4.0
    m.properties["norm"] = 5.0
    m.properties["sum"] = 6.0 + 1.0j
    m.properties["eigenvalues"] = [1.0 + 0.0j, 2.0 + 0.0j]

    path = tmp_path / "props.pycauset"
    try:
        pycauset.save(m, path)
        loaded = pycauset.load(path)
        try:
            props = loaded.properties
            assert props.get("label") == "phase1"
            assert props.get("trace") == pytest.approx(3.0)
            assert props.get("determinant") == pytest.approx(4.0)
            assert props.get("norm") == pytest.approx(5.0)
            assert props.get("sum") == pytest.approx(6.0 + 1.0j)
            eig = props.get("eigenvalues")
            assert eig is not None
            assert [complex(v) for v in eig] == [complex(1.0, 0.0), complex(2.0, 0.0)]
        finally:
            loaded.close()
    finally:
        m.close()


def test_temp_backing_roundtrip_matches_snapshot(tmp_path: Path) -> None:
    orig_threshold = pycauset.get_memory_threshold()
    orig_dir = pycauset._storage_root()

    pycauset.set_backing_dir(tmp_path)
    pycauset.set_memory_threshold(1)

    m = pycauset.TriangularBitMatrix(8)
    m[0, 1] = 1
    m[2, 5] = 1
    path = tmp_path / "temp_roundtrip.pycauset"

    try:
        backing_file = m.get_backing_file()
        assert backing_file is not None and backing_file != ":memory:"
        assert Path(backing_file).resolve().parent == tmp_path.resolve()

        pycauset.save(m, path)
        loaded = pycauset.load(path)
        try:
            assert loaded.rows() == 8
            assert loaded.cols() == 8
            assert loaded[0, 1] == 1
            assert loaded[2, 5] == 1
        finally:
            loaded.close()
    finally:
        m.close()
        pycauset.set_backing_dir(orig_dir)
        if orig_threshold is not None:
            pycauset.set_memory_threshold(orig_threshold)


def test_corrupt_metadata_fails_loudly(tmp_path: Path) -> None:
    m = pycauset.FloatMatrix(2)
    m[0, 0] = 1.0
    m[1, 1] = 2.0
    path = tmp_path / "corrupt.pycauset"

    try:
        pycauset.save(m, path)

        active, _, _ = _persistence._read_active_slot_and_typed_metadata(path)
        meta_offset = int(active["metadata_offset"])
        with path.open("r+b") as f:
            f.seek(meta_offset + 32)
            first = f.read(1)
            if not first:
                pytest.skip("metadata payload unexpectedly empty")
            f.seek(meta_offset + 32)
            f.write(bytes([first[0] ^ 0xFF]))

        with pytest.raises(ValueError):
            pycauset.load(path)
    finally:
        m.close()
        if path.exists():
            try:
                os.remove(path)
            except OSError:
                pass
