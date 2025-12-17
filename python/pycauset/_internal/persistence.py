from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from typing import Any, Protocol


class _HasCopyStorage(Protocol):
    def copy_storage(self, path: str) -> None: ...


def zip_member_data_offset(zip_path: str | Path, info: zipfile.ZipInfo) -> int:
    """Return the absolute byte offset of a zip member's payload.

    This enables memory-mapping data directly out of the zip file.
    """
    zip_path = Path(zip_path)
    with zip_path.open("rb") as f:
        # Local file header layout:
        #   30 bytes fixed header
        #   2 bytes filename length @ offset 26
        #   2 bytes extra length    @ offset 28
        f.seek(info.header_offset + 26)
        n_len, e_len = struct.unpack("<HH", f.read(4))
        return info.header_offset + 30 + n_len + e_len


class PersistenceDeps(Protocol):
    # Types
    CausalSet: type

    TriangularBitMatrix: type | None
    DenseBitMatrix: type | None
    FloatMatrix: type | None
    Float16Matrix: type | None
    Float32Matrix: type | None
    ComplexFloat16Matrix: type | None
    ComplexFloat32Matrix: type | None
    ComplexFloat64Matrix: type | None
    IntegerMatrix: type | None
    Int8Matrix: type | None
    Int16Matrix: type | None
    Int64Matrix: type | None
    UInt8Matrix: type | None
    UInt16Matrix: type | None
    UInt32Matrix: type | None
    UInt64Matrix: type | None
    TriangularFloatMatrix: type | None
    TriangularIntegerMatrix: type | None

    FloatVector: type | None
    Float16Vector: type | None
    Float32Vector: type | None
    ComplexFloat16Vector: type | None
    ComplexFloat32Vector: type | None
    ComplexFloat64Vector: type | None
    Int8Vector: type | None
    IntegerVector: type | None
    Int16Vector: type | None
    Int64Vector: type | None
    UInt8Vector: type | None
    UInt16Vector: type | None
    UInt32Vector: type | None
    UInt64Vector: type | None
    BitVector: type | None
    UnitVector: type | None

    IdentityMatrix: type | None

    # Native module
    native: Any


def save(obj: Any, path: str | Path, *, deps: PersistenceDeps) -> None:
    """Save a matrix or vector to a file (ZIP format)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_raw = path.with_suffix(".raw_tmp")

    # copy_storage now writes raw data to the file
    if hasattr(obj, "copy_storage"):
        obj.copy_storage(str(temp_raw))
    elif isinstance(obj, deps.CausalSet):
        # For CausalSet, save the underlying matrix (current public behavior)
        obj.causal_matrix.copy_storage(str(temp_raw))
        obj = obj.causal_matrix
    else:
        raise TypeError("Object does not support saving (missing copy_storage)")

    try:
        is_transposed = getattr(obj, "is_transposed", False)
        if callable(is_transposed):
            is_transposed = is_transposed()

        is_conjugated = getattr(obj, "is_conjugated", False)
        if callable(is_conjugated):
            is_conjugated = is_conjugated()

        if hasattr(obj, "rows") and hasattr(obj, "cols"):
            rows = obj.rows() if callable(obj.rows) else obj.rows
            cols = obj.cols() if callable(obj.cols) else obj.cols
        elif hasattr(obj, "size"):
            rows = obj.size()
            cols = 1
        else:
            rows = len(obj)
            cols = 1

        metadata: dict[str, Any] = {
            "rows": rows,
            "cols": cols,
            "seed": getattr(obj, "seed", 0),
            "is_transposed": is_transposed,
            "is_conjugated": is_conjugated,
        }

        scalar = getattr(obj, "scalar", 1.0)
        if isinstance(scalar, complex):
            metadata["scalar"] = {"real": scalar.real, "imag": scalar.imag}
        else:
            metadata["scalar"] = scalar

        # Determine matrix_type and data_type based on class
        if deps.TriangularBitMatrix is not None and isinstance(obj, deps.TriangularBitMatrix):
            metadata["matrix_type"] = "CAUSAL"
            metadata["data_type"] = "BIT"
        elif deps.DenseBitMatrix is not None and isinstance(obj, deps.DenseBitMatrix):
            metadata["matrix_type"] = "DENSE_FLOAT"
            metadata["data_type"] = "BIT"
        elif deps.Float16Matrix is not None and isinstance(obj, deps.Float16Matrix):
            metadata["matrix_type"] = "DENSE_FLOAT"
            metadata["data_type"] = "FLOAT16"
        elif deps.Float32Matrix is not None and isinstance(obj, deps.Float32Matrix):
            metadata["matrix_type"] = "DENSE_FLOAT"
            metadata["data_type"] = "FLOAT32"
        elif deps.FloatMatrix is not None and isinstance(obj, deps.FloatMatrix):
            metadata["matrix_type"] = "DENSE_FLOAT"
            metadata["data_type"] = "FLOAT64"
        elif deps.ComplexFloat16Matrix is not None and isinstance(obj, deps.ComplexFloat16Matrix):
            metadata["matrix_type"] = "DENSE_FLOAT"
            metadata["data_type"] = "COMPLEX_FLOAT16"
        elif deps.ComplexFloat32Matrix is not None and isinstance(obj, deps.ComplexFloat32Matrix):
            metadata["matrix_type"] = "DENSE_FLOAT"
            metadata["data_type"] = "COMPLEX_FLOAT32"
        elif deps.ComplexFloat64Matrix is not None and isinstance(obj, deps.ComplexFloat64Matrix):
            metadata["matrix_type"] = "DENSE_FLOAT"
            metadata["data_type"] = "COMPLEX_FLOAT64"
        elif deps.Int8Matrix is not None and isinstance(obj, deps.Int8Matrix):
            metadata["matrix_type"] = "INTEGER"
            metadata["data_type"] = "INT8"
        elif deps.IntegerMatrix is not None and isinstance(obj, deps.IntegerMatrix):
            metadata["matrix_type"] = "INTEGER"
            metadata["data_type"] = "INT32"
        elif deps.Int16Matrix is not None and isinstance(obj, deps.Int16Matrix):
            metadata["matrix_type"] = "INTEGER"
            metadata["data_type"] = "INT16"
        elif deps.Int64Matrix is not None and isinstance(obj, deps.Int64Matrix):
            metadata["matrix_type"] = "INTEGER"
            metadata["data_type"] = "INT64"
        elif deps.UInt8Matrix is not None and isinstance(obj, deps.UInt8Matrix):
            metadata["matrix_type"] = "INTEGER"
            metadata["data_type"] = "UINT8"
        elif deps.UInt16Matrix is not None and isinstance(obj, deps.UInt16Matrix):
            metadata["matrix_type"] = "INTEGER"
            metadata["data_type"] = "UINT16"
        elif deps.UInt32Matrix is not None and isinstance(obj, deps.UInt32Matrix):
            metadata["matrix_type"] = "INTEGER"
            metadata["data_type"] = "UINT32"
        elif deps.UInt64Matrix is not None and isinstance(obj, deps.UInt64Matrix):
            metadata["matrix_type"] = "INTEGER"
            metadata["data_type"] = "UINT64"
        elif deps.TriangularFloatMatrix is not None and isinstance(obj, deps.TriangularFloatMatrix):
            metadata["matrix_type"] = "TRIANGULAR_FLOAT"
            metadata["data_type"] = "FLOAT64"
        elif deps.TriangularIntegerMatrix is not None and isinstance(obj, deps.TriangularIntegerMatrix):
            metadata["matrix_type"] = "TRIANGULAR_INTEGER"
            metadata["data_type"] = "INT32"
        elif deps.FloatVector is not None and isinstance(obj, deps.FloatVector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "FLOAT64"
        elif deps.Float32Vector is not None and isinstance(obj, deps.Float32Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "FLOAT32"
        elif deps.Float16Vector is not None and isinstance(obj, deps.Float16Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "FLOAT16"
        elif deps.ComplexFloat16Vector is not None and isinstance(obj, deps.ComplexFloat16Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "COMPLEX_FLOAT16"
        elif deps.ComplexFloat32Vector is not None and isinstance(obj, deps.ComplexFloat32Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "COMPLEX_FLOAT32"
        elif deps.ComplexFloat64Vector is not None and isinstance(obj, deps.ComplexFloat64Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "COMPLEX_FLOAT64"
        elif deps.Int8Vector is not None and isinstance(obj, deps.Int8Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "INT8"
        elif deps.IntegerVector is not None and isinstance(obj, deps.IntegerVector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "INT32"
        elif deps.Int16Vector is not None and isinstance(obj, deps.Int16Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "INT16"
        elif deps.Int64Vector is not None and isinstance(obj, deps.Int64Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "INT64"
        elif deps.UInt8Vector is not None and isinstance(obj, deps.UInt8Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "UINT8"
        elif deps.UInt16Vector is not None and isinstance(obj, deps.UInt16Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "UINT16"
        elif deps.UInt32Vector is not None and isinstance(obj, deps.UInt32Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "UINT32"
        elif deps.UInt64Vector is not None and isinstance(obj, deps.UInt64Vector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "UINT64"
        elif deps.BitVector is not None and isinstance(obj, deps.BitVector):
            metadata["matrix_type"] = "VECTOR"
            metadata["data_type"] = "BIT"
        elif deps.UnitVector is not None and isinstance(obj, deps.UnitVector):
            metadata["matrix_type"] = "UNIT_VECTOR"
            metadata["data_type"] = "FLOAT64"
        elif deps.IdentityMatrix is not None and isinstance(obj, deps.IdentityMatrix):
            metadata["matrix_type"] = "IDENTITY"
            metadata["data_type"] = "FLOAT64"

        if hasattr(obj, "cached_trace") and obj.cached_trace is not None:
            metadata["cached_trace"] = obj.cached_trace

        if hasattr(obj, "cached_determinant") and obj.cached_determinant is not None:
            metadata["cached_determinant"] = obj.cached_determinant

        if hasattr(obj, "cached_eigenvalues") and obj.cached_eigenvalues is not None:
            metadata["cached_eigenvalues"] = [[z.real, z.imag] for z in obj.cached_eigenvalues]

        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            zf.write(temp_raw, "data.bin")

    finally:
        if temp_raw.exists():
            try:
                temp_raw.unlink()
            except OSError:
                pass


def load(path: str | Path, *, deps: PersistenceDeps) -> Any:
    """Load a matrix or vector from a file (ZIP format)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("metadata.json") as f:
            metadata = json.load(f)

        try:
            with zf.open("cache.json") as f:
                cache_meta = json.load(f)
                metadata.update(cache_meta)
        except KeyError:
            pass

        info = zf.getinfo("data.bin")
        data_offset = zip_member_data_offset(path, info)

    matrix_type = metadata.get("matrix_type")
    data_type = metadata.get("data_type")
    rows = metadata.get("rows", 0)
    cols = metadata.get("cols", 0)
    seed = metadata.get("seed", 0)
    scalar = metadata.get("scalar", 1.0)
    if isinstance(scalar, dict) and "real" in scalar and "imag" in scalar:
        scalar = complex(scalar["real"], scalar["imag"])
    is_transposed = metadata.get("is_transposed", False)
    is_conjugated = metadata.get("is_conjugated", False)

    def _require_square_dims_for_type(type_name: str) -> None:
        # Back-compat: older files may omit cols (stored as 0 by default).
        # If cols is present and non-zero, it must match rows for square-only types.
        if cols not in (0, rows):
            raise ValueError(f"{type_name} requires rows == cols (got rows={rows}, cols={cols})")

    obj = None
    if matrix_type == "CAUSAL" and deps.TriangularBitMatrix is not None:
        _require_square_dims_for_type("CAUSAL")
        obj = deps.TriangularBitMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "IDENTITY" and deps.IdentityMatrix is not None:
        matrix_cols = cols or rows
        try:
            obj = deps.IdentityMatrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
        except TypeError:
            # Backward-compat: older extensions only supported square identity.
            obj = deps.IdentityMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "DENSE_FLOAT":
        matrix_cols = cols or rows
        if data_type == "BIT" and deps.DenseBitMatrix is not None:
            try:
                obj = deps.DenseBitMatrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
            except TypeError:
                obj = deps.DenseBitMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT16" and deps.Float16Matrix is not None:
            try:
                obj = deps.Float16Matrix._from_storage(
                    rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed
                )
            except TypeError:
                obj = deps.Float16Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT32" and deps.Float32Matrix is not None:
            try:
                obj = deps.Float32Matrix._from_storage(
                    rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed
                )
            except TypeError:
                obj = deps.Float32Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT64" and deps.FloatMatrix is not None:
            try:
                obj = deps.FloatMatrix._from_storage(
                    rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed
                )
            except TypeError:
                obj = deps.FloatMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT16" and deps.ComplexFloat16Matrix is not None:
            try:
                obj = deps.ComplexFloat16Matrix._from_storage(
                    rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed
                )
            except TypeError:
                obj = deps.ComplexFloat16Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT32" and deps.ComplexFloat32Matrix is not None:
            try:
                obj = deps.ComplexFloat32Matrix._from_storage(
                    rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed
                )
            except TypeError:
                obj = deps.ComplexFloat32Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT64" and deps.ComplexFloat64Matrix is not None:
            try:
                obj = deps.ComplexFloat64Matrix._from_storage(
                    rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed
                )
            except TypeError:
                obj = deps.ComplexFloat64Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "INTEGER":
        matrix_cols = cols or rows
        if data_type == "INT8" and deps.Int8Matrix is not None:
            try:
                obj = deps.Int8Matrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
            except TypeError:
                obj = deps.Int8Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT16" and deps.Int16Matrix is not None:
            try:
                obj = deps.Int16Matrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
            except TypeError:
                obj = deps.Int16Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT32" and deps.IntegerMatrix is not None:
            try:
                obj = deps.IntegerMatrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
            except TypeError:
                obj = deps.IntegerMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT64" and deps.Int64Matrix is not None:
            try:
                obj = deps.Int64Matrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
            except TypeError:
                obj = deps.Int64Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT8" and deps.UInt8Matrix is not None:
            try:
                obj = deps.UInt8Matrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
            except TypeError:
                obj = deps.UInt8Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT16" and deps.UInt16Matrix is not None:
            try:
                obj = deps.UInt16Matrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
            except TypeError:
                obj = deps.UInt16Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT32" and deps.UInt32Matrix is not None:
            try:
                obj = deps.UInt32Matrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
            except TypeError:
                obj = deps.UInt32Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT64" and deps.UInt64Matrix is not None:
            try:
                obj = deps.UInt64Matrix._from_storage(rows, matrix_cols, str(path), data_offset, seed, scalar, is_transposed)
            except TypeError:
                obj = deps.UInt64Matrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "TRIANGULAR_FLOAT" and deps.TriangularFloatMatrix is not None:
        _require_square_dims_for_type("TRIANGULAR_FLOAT")
        obj = deps.TriangularFloatMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "TRIANGULAR_INTEGER" and deps.TriangularIntegerMatrix is not None:
        _require_square_dims_for_type("TRIANGULAR_INTEGER")
        obj = deps.TriangularIntegerMatrix._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "VECTOR":
        if data_type == "FLOAT64" and deps.FloatVector is not None:
            obj = deps.FloatVector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT32" and deps.Float32Vector is not None:
            obj = deps.Float32Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "FLOAT16" and deps.Float16Vector is not None:
            obj = deps.Float16Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT16" and deps.ComplexFloat16Vector is not None:
            obj = deps.ComplexFloat16Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT32" and deps.ComplexFloat32Vector is not None:
            obj = deps.ComplexFloat32Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "COMPLEX_FLOAT64" and deps.ComplexFloat64Vector is not None:
            obj = deps.ComplexFloat64Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT8" and deps.Int8Vector is not None:
            obj = deps.Int8Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT32" and deps.IntegerVector is not None:
            obj = deps.IntegerVector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT16" and deps.Int16Vector is not None:
            obj = deps.Int16Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "INT64" and deps.Int64Vector is not None:
            obj = deps.Int64Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT8" and deps.UInt8Vector is not None:
            obj = deps.UInt8Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT16" and deps.UInt16Vector is not None:
            obj = deps.UInt16Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT32" and deps.UInt32Vector is not None:
            obj = deps.UInt32Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "UINT64" and deps.UInt64Vector is not None:
            obj = deps.UInt64Vector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
        elif data_type == "BIT" and deps.BitVector is not None:
            obj = deps.BitVector._from_storage(rows, str(path), data_offset, seed, scalar, is_transposed)
    elif matrix_type == "UNIT_VECTOR" and deps.UnitVector is not None:
        obj = deps.UnitVector._from_storage(rows, seed, str(path), data_offset, seed, scalar, is_transposed)

    if obj is None:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

    if is_transposed and hasattr(obj, "set_transposed"):
        try:
            obj.set_transposed(True)
        except Exception:
            pass

    if is_conjugated and hasattr(obj, "set_conjugated"):
        try:
            obj.set_conjugated(True)
        except Exception:
            pass

    cached_trace = metadata.get("cached_trace")
    if cached_trace is not None:
        try:
            obj.cached_trace = cached_trace
        except Exception:
            pass

    cached_determinant = metadata.get("cached_determinant")
    if cached_determinant is not None:
        try:
            obj.cached_determinant = cached_determinant
        except Exception:
            pass

    cached_eigenvalues = metadata.get("cached_eigenvalues")
    if cached_eigenvalues is not None:
        try:
            obj.cached_eigenvalues = [complex(r, i) for r, i in cached_eigenvalues]
        except Exception:
            pass

    if metadata.get("object_type") == "CausalSet":
        st_meta = metadata.get("spacetime", {})
        st_type = st_meta.get("type")
        st_args = st_meta.get("args", {})

        st = None
        if st_type == "MinkowskiDiamond":
            st = deps.native.MinkowskiDiamond(st_args.get("dimension", 2))
        elif st_type == "MinkowskiCylinder":
            st = deps.native.MinkowskiCylinder(
                st_args.get("dimension", 2),
                st_args.get("height", 1.0),
                st_args.get("circumference", 1.0),
            )
        elif st_type == "MinkowskiBox":
            st = deps.native.MinkowskiBox(
                st_args.get("dimension", 2),
                st_args.get("time_extent", 1.0),
                st_args.get("space_extent", 1.0),
            )

        return deps.CausalSet(n=rows, spacetime=st, seed=seed, matrix=obj)

    return obj
