from __future__ import annotations

from pathlib import Path

import pycauset
from pycauset._internal.blockmatrix import BlockMatrix, block_matmul
from pycauset._internal.thunks import StaleThunkError


def test_blockmatrix_save_load_roundtrip(tmp_path: Path):
    A = pycauset.matrix([[1, 2], [3, 4]])
    B = pycauset.matrix([[5], [6]])
    C = pycauset.matrix([[7, 8]])
    D = pycauset.matrix([[9]])

    M = BlockMatrix([[A, B], [C, D]])  # 3x3

    p = tmp_path / "bm.pycauset"
    pycauset.save(M, p)

    M2 = pycauset.load(p)
    assert isinstance(M2, BlockMatrix)
    assert M2.shape == (3, 3)

    for i in range(3):
        for j in range(3):
            assert M2.get(i, j) == M.get(i, j)


def test_blockmatrix_save_load_roundtrip_nested(tmp_path: Path):
    inner = BlockMatrix([[pycauset.matrix([[1, 2], [3, 4]])]])
    outer = BlockMatrix([[inner]])

    p = tmp_path / "bm_nested.pycauset"
    pycauset.save(outer, p)

    out = pycauset.load(p)
    assert isinstance(out, BlockMatrix)
    assert out.shape == (2, 2)
    assert out.get(0, 0) == 1
    assert out.get(0, 1) == 2
    assert out.get(1, 0) == 3
    assert out.get(1, 1) == 4

    # Ensure nested sidecar directory exists for the child BlockMatrix file.
    top_blocks = Path(str(p) + ".blocks")
    child = top_blocks / "block_r0_c0.pycauset"
    assert child.exists()
    child_blocks = Path(str(child) + ".blocks")
    assert child_blocks.exists()
    assert (child_blocks / "block_r0_c0.pycauset").exists()


def test_blockmatrix_save_evaluates_thunks_blockwise(tmp_path: Path):
    # 1x2 @ 2x1 -> 1x1 output thunk block
    A0 = pycauset.matrix([[1]])
    A1 = pycauset.matrix([[2]])
    B0 = pycauset.matrix([[3]])
    B1 = pycauset.matrix([[4]])

    left = BlockMatrix([[A0, A1]])
    right = BlockMatrix([[B0], [B1]])

    out = block_matmul(left, right)

    p = tmp_path / "bm_thunk.pycauset"
    pycauset.save(out, p)

    out2 = pycauset.load(p)
    assert isinstance(out2, BlockMatrix)
    assert out2.get(0, 0) == (1 * 3) + (2 * 4)

    # Ensure child block files exist (deterministic naming).
    blocks_dir = Path(str(p) + ".blocks")
    assert (blocks_dir / "block_r0_c0.pycauset").exists()


def test_blockmatrix_save_raises_on_stale_thunks(tmp_path: Path):
    A0 = pycauset.matrix([[1]])
    A1 = pycauset.matrix([[2]])
    B0 = pycauset.matrix([[3]])
    B1 = pycauset.matrix([[4]])

    left = BlockMatrix([[A0, A1]])
    right = BlockMatrix([[B0], [B1]])
    out = block_matmul(left, right)

    # Bump version after thunk creation -> stale at save time.
    left.set_block(0, 0, A0)

    p = tmp_path / "bm_stale.pycauset"
    try:
        pycauset.save(out, p)
        raise AssertionError("expected stale thunk save to raise")
    except StaleThunkError:
        pass


def test_blockmatrix_overwrite_cleanup_preserves_unrelated_files(tmp_path: Path):
    A = pycauset.matrix([[1]])
    M = BlockMatrix([[A]])

    p = tmp_path / "bm_overwrite.pycauset"
    pycauset.save(M, p)

    blocks_dir = Path(str(p) + ".blocks")
    keep = blocks_dir / "KEEP_ME.txt"
    keep.write_text("do not delete")

    pycauset.save(M, p)
    assert keep.exists()


def test_blockmatrix_load_detects_child_replacement_via_payload_uuid(tmp_path: Path):
    A = pycauset.matrix([[1, 2], [3, 4]])
    M = BlockMatrix([[A]])

    p = tmp_path / "bm_uuid_pin.pycauset"
    pycauset.save(M, p)

    blocks_dir = Path(str(p) + ".blocks")
    child = blocks_dir / "block_r0_c0.pycauset"
    assert child.exists()

    # Replace the child file with different contents at the same path.
    A2 = pycauset.matrix([[9, 9], [9, 9]])
    pycauset.save(A2, child)

    try:
        pycauset.load(p)
        raise AssertionError("expected payload_uuid mismatch to raise")
    except ValueError as e:
        assert "payload_uuid mismatch" in str(e)


def test_blockmatrix_save_load_roundtrip_mixed_dtypes(tmp_path: Path):
    # Mixed leaf dtypes should round-trip per-child.
    A = pycauset.matrix([[1, 2], [3, 4]], dtype="int32")
    B = pycauset.matrix([[1.5, 2.5], [3.5, 4.5]], dtype="float32")

    M = BlockMatrix([[A, B]])
    p = tmp_path / "bm_mixed.pycauset"
    pycauset.save(M, p)

    M2 = pycauset.load(p)
    assert isinstance(M2, BlockMatrix)
    assert M2.shape == (2, 4)
    assert M2.get(0, 0) == 1
    assert float(M2.get(0, 2)) == float(B.get(0, 0))

    a2 = M2.get_block(0, 0)
    b2 = M2.get_block(0, 1)
    assert getattr(a2, "dtype", None) == getattr(A, "dtype", None)
    assert getattr(b2, "dtype", None) == getattr(B, "dtype", None)


def test_blockmatrix_save_load_roundtrip_with_view_blocks(tmp_path: Path):
    base = pycauset.matrix([[1, 2], [3, 4]], dtype="float32")
    M = BlockMatrix([[base]]).refine_partitions(row_partitions=[0, 1, 2], col_partitions=[0, 1, 2])

    # Refined grid contains SubmatrixView blocks; save should materialize each view block locally.
    p = tmp_path / "bm_view_blocks.pycauset"
    pycauset.save(M, p)

    M2 = pycauset.load(p)
    assert isinstance(M2, BlockMatrix)
    assert M2.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            assert float(M2.get(i, j)) == float(base.get(i, j))


def test_view_blocks_keep_their_dtype_and_values(tmp_path):
    """3.9 C-5: view blocks were copied element by element into a NumPy array whose dtype came
    from a hint table (complex_float16 came back complex_float32)."""
    import numpy as np

    import pycauset as pc
    from pycauset._internal.blockmatrix import BlockMatrix
    from pycauset._internal.submatrix_view import SubmatrixView

    dense = pc.matrix(np.arange(36.0).reshape(6, 6))
    c16 = pc.zeros((4, 4), dtype="complex_float16")
    c16.set(1, 1, 2 + 1j)
    for src, want in ((dense, "float64"), (c16, "complex_float16")):
        bm = BlockMatrix([[SubmatrixView(src, 1, 1, 2, 2)]])
        pc.save(bm, tmp_path / f"b_{want}.pycauset")
        back = pc.load(tmp_path / f"b_{want}.pycauset")
        leaf = back.get_block(0, 0) if hasattr(back, "get_block") else back._blocks[0][0]
        assert str(leaf.dtype) == want
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(src)[1:3, 1:3])
