#pragma once
// Indexing of matrices from Python (review PLAN 4.2: moved out of bind_matrix.cpp unchanged):
// subscript parsing (integers, slices, index arrays), dense slicing and assignment,
// element get / set through every view, and the MatrixBase __getitem__ / __setitem__
// dispatch. Used only by bind_matrix.cpp.
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/math/VectorOps.hpp"
#include "pycauset/math/ViewValue.hpp"
#include <tuple>
#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/math/Bitwise.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include <cmath>
#include <limits>
#include "bindings_common.hpp"
#include "element_conversion.hpp"
#include "scalar_args.hpp"
#include "pycauset/math/Reductions.hpp"
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <iostream>

#include "binding_warnings.hpp"

#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/core/ObjectFactory.hpp"

#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/VectorBase.hpp"

#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/ParallelUtils.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "numpy_import.hpp"


namespace pycauset_bindings::indexing {

using namespace pycauset;
using namespace pycauset_bindings::numpy_import;

struct SliceInfo {
    bool is_scalar;
    bool is_array;
    uint64_t index;
    py::ssize_t start;
    py::ssize_t step;
    uint64_t length;
    std::vector<uint64_t> indices;
};

struct ParsedIndex {
    SliceInfo rows;
    SliceInfo cols;
};

using pycauset_bindings::coerce_bit_value;

inline SliceInfo make_scalar_index(py::ssize_t idx, uint64_t dim) {
    if (idx < 0) {
        idx += static_cast<py::ssize_t>(dim);
    }
    if (idx < 0 || idx >= static_cast<py::ssize_t>(dim)) {
        throw py::index_error("Index out of bounds");
    }
    return SliceInfo{true, false, static_cast<uint64_t>(idx), 0, 1, 1, {}};
}

inline SliceInfo make_full_slice(uint64_t dim) {
    return SliceInfo{false, false, 0, 0, 1, dim, {}};
}

inline SliceInfo slice_from_object(const py::handle& obj, uint64_t dim) {
    if (py::isinstance<py::int_>(obj)) {
        return make_scalar_index(obj.cast<py::ssize_t>(), dim);
    }
    // NumPy integer scalars index like ints (they implement __index__; bool does not count).
    if (!PyBool_Check(obj.ptr()) && !py::isinstance<py::array>(obj) && PyIndex_Check(obj.ptr())) {
        const py::ssize_t idx = PyNumber_AsSsize_t(obj.ptr(), PyExc_IndexError);
        if (idx == -1 && PyErr_Occurred()) throw py::error_already_set();
        return make_scalar_index(idx, dim);
    }
    // A Python list is an index array, as in NumPy (m[[0, 2], :]).
    if (py::isinstance<py::list>(obj)) {
        return slice_from_object(py::array::ensure(obj), dim);
    }
    if (py::isinstance<py::slice>(obj)) {
        py::slice s = obj.cast<py::slice>();
        py::ssize_t start, stop, step, slicelength;
        if (!s.compute(static_cast<py::ssize_t>(dim), &start, &stop, &step, &slicelength)) {
            throw py::value_error("Invalid slice");
        }
        return SliceInfo{false, false, 0, start, step, static_cast<uint64_t>(slicelength), {}};
    }
    if (py::isinstance<py::ellipsis>(obj)) {
        return make_full_slice(dim);
    }
    if (obj.is_none()) {
        throw py::value_error("newaxis/None would create >2D result; not supported (matrices/vectors only)");
    }
    if (py::isinstance<py::array>(obj)) {
        if (py::isinstance<py::array_t<bool>>(obj)) {
            auto mask = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(obj);
            if (!mask) {
                throw py::type_error("Boolean index array must be 1D bool");
            }
            auto buf = mask.request();
            if (buf.ndim != 1) {
                throw py::type_error("Boolean index array must be 1D");
            }
            if (static_cast<uint64_t>(buf.shape[0]) != dim) {
                throw py::index_error("Boolean index length must match dimension length");
            }
            const bool* ptr = static_cast<const bool*>(buf.ptr);
            std::vector<uint64_t> indices;
            indices.reserve(static_cast<size_t>(buf.shape[0]));
            for (py::ssize_t i = 0; i < buf.shape[0]; ++i) {
                if (ptr[i]) {
                    indices.push_back(static_cast<uint64_t>(i));
                }
            }
            return SliceInfo{false, true, 0, 0, 1, static_cast<uint64_t>(indices.size()), std::move(indices)};
        }

        auto idx = py::array_t<long long, py::array::c_style | py::array::forcecast>::ensure(obj);
        if (!idx) {
            throw py::type_error("Advanced index arrays must be integer or boolean");
        }
        auto buf = idx.request();
        if (buf.ndim != 1) {
            throw py::type_error("Advanced index array must be 1D");
        }
        const long long* ptr = static_cast<const long long*>(buf.ptr);
        std::vector<uint64_t> indices;
        indices.reserve(static_cast<size_t>(buf.shape[0]));
        for (py::ssize_t i = 0; i < buf.shape[0]; ++i) {
            long long v = ptr[i];
            if (v < 0) {
                v += static_cast<long long>(dim);
            }
            if (v < 0 || v >= static_cast<long long>(dim)) {
                throw py::index_error("Index out of bounds");
            }
            indices.push_back(static_cast<uint64_t>(v));
        }
        return SliceInfo{false, true, 0, 0, 1, static_cast<uint64_t>(indices.size()), std::move(indices)};
    }
    throw py::type_error("Matrix indices must be integers or slices");
}

inline ParsedIndex parse_matrix_subscript(const py::handle& key, uint64_t rows, uint64_t cols) {
    // Normalize to exactly two entries (row, col). Ellipsis expands to full slices: a real
    // slice(None) object (pybind11's default py::slice() is a null handle, so `m[...]` raised).
    const auto full = [] { return py::reinterpret_steal<py::object>(PySlice_New(nullptr, nullptr, nullptr)); };
    py::object first;
    py::object second;

    if (py::isinstance<py::ellipsis>(key)) {
        first = full();
        second = full();
    } else if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
        py::sequence seq = key.cast<py::sequence>();
        if (seq.size() == 1) {
            if (py::isinstance<py::ellipsis>(seq[0])) {
                first = full();
                second = full();
            } else {
                throw py::type_error("Matrix indices must be (row, col)");
            }
        } else if (seq.size() == 2) {
            first = seq[0];
            second = seq[1];
            if (py::isinstance<py::ellipsis>(first)) {
                first = full();
            }
            if (py::isinstance<py::ellipsis>(second)) {
                second = full();
            }
        } else {
            throw py::type_error("Matrix indices must be 2D (row, col) for matrices/vectors");
        }
    } else {
        throw py::type_error("Matrix indices must be provided as [row, col]");
    }

    SliceInfo r = slice_from_object(first, rows);
    SliceInfo c = slice_from_object(second, cols);
    return ParsedIndex{r, c};
}

inline uint64_t slice_length(const SliceInfo& s) {
    return s.is_scalar ? 1 : s.length;
}

inline std::pair<uint64_t, uint64_t> slice_shape(const SliceInfo& r, const SliceInfo& c) {
    uint64_t out_r = slice_length(r);
    uint64_t out_c = slice_length(c);
    return {out_r, out_c};
}

template <typename T>
inline std::shared_ptr<DenseMatrix<T>> slice_dense_matrix(const DenseMatrix<T>& mat, const SliceInfo& r, const SliceInfo& c) {
    const bool has_array = r.is_array || c.is_array;
    // The copies below hold stored values (get() applies the transposition and window); they
    // keep the source's factor and conjugation, or B[[0, 1], [2, 3]] of B = A * 2 read A's values.
    auto keep_view = [&](std::shared_ptr<DenseMatrix<T>> out) {
        out->set_scalar(mat.get_scalar());
        out->set_conjugated(mat.is_conjugated());
        return out;
    };

    // DenseMatrix<bool> (DenseBitMatrix) does not support the mapper/view-offset constructor.
    // Guard the view path at compile time so the bool instantiation still compiles.
    if constexpr (!std::is_same_v<T, bool>) {
        const bool can_view = !has_array && r.step == 1 && c.step == 1;
        if (can_view) {
            const uint64_t logical_rows = mat.is_transposed() ? slice_length(c) : slice_length(r);
            const uint64_t logical_cols = mat.is_transposed() ? slice_length(r) : slice_length(c);

            const uint64_t row_start = r.is_scalar ? r.index : static_cast<uint64_t>(r.start);
            const uint64_t col_start = c.is_scalar ? c.index : static_cast<uint64_t>(c.start);
            const uint64_t base_row_offset = mat.row_offset() + (mat.is_transposed() ? col_start : row_start);
            const uint64_t base_col_offset = mat.col_offset() + (mat.is_transposed() ? row_start : col_start);

            auto view = std::make_shared<DenseMatrix<T>>(logical_rows,
                                                         logical_cols,
                                                         mat.shared_mapper(),
                                                         mat.base_rows(),
                                                         mat.base_cols(),
                                                         base_row_offset,
                                                         base_col_offset,
                                                         mat.get_seed(),
                                                         mat.get_scalar(),
                                                         mat.is_transposed(),
                                                         mat.is_conjugated());
            return view;
        }
    }

    if (r.is_array && c.is_array) {
        const uint64_t len_r = slice_length(r);
        const uint64_t len_c = slice_length(c);
        uint64_t out_len = 0;
        if (len_r == len_c) {
            out_len = len_r;
        } else if (len_r == 1) {
            out_len = len_c;
        } else if (len_c == 1) {
            out_len = len_r;
        } else {
            throw py::value_error("Row and column index arrays must have the same length or be broadcastable (length 1)");
        }

        auto result = std::make_shared<DenseMatrix<T>>(1, out_len);
        for (uint64_t k = 0; k < out_len; ++k) {
            const uint64_t src_r = r.indices[len_r == 1 ? 0 : k];
            const uint64_t src_c = c.indices[len_c == 1 ? 0 : k];
            result->set(0, k, mat.get(src_r, src_c));
        }
        return keep_view(result);
    }

    if (r.is_array) {
        const uint64_t out_r = slice_length(r);
        const uint64_t out_c = slice_length(c);
        auto result = std::make_shared<DenseMatrix<T>>(out_r, out_c);
        for (uint64_t rr = 0; rr < out_r; ++rr) {
            const uint64_t src_r = r.indices[rr];
            for (uint64_t cc = 0; cc < out_c; ++cc) {
                const uint64_t src_c = c.is_scalar ? c.index : static_cast<uint64_t>(c.start + static_cast<py::ssize_t>(cc) * c.step);
                result->set(rr, cc, mat.get(src_r, src_c));
            }
        }
        return keep_view(result);
    }

    if (c.is_array) {
        const uint64_t out_r = slice_length(r);
        const uint64_t out_c = slice_length(c);
        auto result = std::make_shared<DenseMatrix<T>>(out_r, out_c);
        for (uint64_t rr = 0; rr < out_r; ++rr) {
            const uint64_t src_r = r.is_scalar ? r.index : static_cast<uint64_t>(r.start + static_cast<py::ssize_t>(rr) * r.step);
            for (uint64_t cc = 0; cc < out_c; ++cc) {
                const uint64_t src_c = c.indices[cc];
                result->set(rr, cc, mat.get(src_r, src_c));
            }
        }
        return keep_view(result);
    }

    auto [out_r, out_c] = slice_shape(r, c);
    auto result = std::make_shared<DenseMatrix<T>>(out_r, out_c);
    for (uint64_t rr = 0; rr < out_r; ++rr) {
        const uint64_t src_r = r.is_scalar ? r.index : static_cast<uint64_t>(r.start + static_cast<py::ssize_t>(rr) * r.step);
        for (uint64_t cc = 0; cc < out_c; ++cc) {
            const uint64_t src_c = c.is_scalar ? c.index : static_cast<uint64_t>(c.start + static_cast<py::ssize_t>(cc) * c.step);
            result->set(rr, cc, mat.get(src_r, src_c));
        }
    }
    return keep_view(result);
}

template <typename T>
inline T cast_scalar_to(const py::object& value) {
    return value.cast<T>();
}

template <>
inline bool cast_scalar_to<bool>(const py::object& value) {
    return coerce_bit_value(value);
}

template <typename T>
inline void maybe_warn_assignment_cast(const py::dtype& src_dtype) {
    const py::dtype target_dtype = py::dtype::of<T>();
    if (src_dtype.equal(target_dtype)) {
        return;
    }

    const auto src_name = py::str(static_cast<py::object>(src_dtype)).cast<std::string>();
    const auto tgt_name = py::str(static_cast<py::object>(target_dtype)).cast<std::string>();
    std::string msg = "pycauset assignment: casting RHS from ";
    msg += src_name;
    msg += " to ";
    msg += tgt_name;
    bindings_warn::warn_with_category(msg,
        "PyCausetDTypeWarning",
        /*stacklevel=*/3);

    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        const py::ssize_t src_size = src_dtype.itemsize();
        const py::ssize_t tgt_size = static_cast<py::ssize_t>(sizeof(T));
        const char kind = src_dtype.kind();
        if (kind == 'f' || src_size > tgt_size) {
            std::string risk = "pycauset assignment: possible overflow when casting from ";
            risk += src_name;
            risk += " to ";
            risk += tgt_name;
            bindings_warn::warn_with_category(risk,
                "PyCausetOverflowRiskWarning",
                /*stacklevel=*/3);
        }
    }
}

template <typename T>
struct AssignmentValue {
    bool is_scalar = true;
    T scalar{};
    std::shared_ptr<DenseMatrix<T>> mat;
    uint64_t rows = 1;
    uint64_t cols = 1;
};

template <typename T>
inline AssignmentValue<T> normalize_assignment_value(const py::object& value) {
    AssignmentValue<T> out;

    if (py::isinstance<py::array>(value)) {
        py::array arr = py::array::ensure(value);
        const py::dtype src_dtype = arr.dtype();
        auto buf = arr.request();
        if (buf.ndim == 0) {
            maybe_warn_assignment_cast<T>(src_dtype);
            out.scalar = value.cast<T>();
            out.is_scalar = true;
            return out;
        }
        if (buf.ndim == 1 || buf.ndim == 2) {
            const uint64_t rows = (buf.ndim == 1) ? 1 : static_cast<uint64_t>(buf.shape[0]);
            const uint64_t cols = (buf.ndim == 1) ? static_cast<uint64_t>(buf.shape[0]) : static_cast<uint64_t>(buf.shape[1]);
            py::array_t<T, py::array::c_style | py::array::forcecast> arr_t = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(arr);
            if (!arr_t) {
                throw py::type_error("Assignment array dtype is not convertible to target matrix dtype");
            }
            maybe_warn_assignment_cast<T>(src_dtype);
            auto converted = arr_t.request();
            auto m = std::make_shared<DenseMatrix<T>>(rows, cols);
            const T* src = static_cast<const T*>(converted.ptr);
            const uint64_t stride0 = static_cast<uint64_t>(converted.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
            const uint64_t stride1 = (converted.ndim == 1)
                                         ? 1
                                         : static_cast<uint64_t>(converted.strides[1] / static_cast<py::ssize_t>(sizeof(T)));
            for (uint64_t i = 0; i < rows; ++i) {
                for (uint64_t j = 0; j < cols; ++j) {
                    m->set(i, j, src[i * stride0 + j * stride1]);
                }
            }
            out.is_scalar = false;
            out.mat = std::move(m);
            out.rows = rows;
            out.cols = cols;
            return out;
        }
        throw py::value_error("Assignment array rank must be 0, 1, or 2");
    }

    if (py::isinstance<std::shared_ptr<DenseMatrix<T>>>(value)) {
        out.mat = value.cast<std::shared_ptr<DenseMatrix<T>>>();
        out.is_scalar = false;
        out.rows = out.mat->rows();
        out.cols = out.mat->cols();
        return out;
    }

    out.scalar = cast_scalar_to<T>(value);
    out.is_scalar = true;
    return out;
}

template <typename T>
inline bool can_broadcast_to(uint64_t target_rows, uint64_t target_cols, const AssignmentValue<T>& v) {
    if (v.is_scalar) {
        return true;
    }
    const bool rows_ok = (v.rows == target_rows) || (v.rows == 1);
    const bool cols_ok = (v.cols == target_cols) || (v.cols == 1);
    return rows_ok && cols_ok;
}

template <typename T>
inline T value_at(const AssignmentValue<T>& v, uint64_t i, uint64_t j) {
    if (v.is_scalar) {
        return v.scalar;
    }
    const uint64_t src_i = (v.rows == 1) ? 0 : i;
    const uint64_t src_j = (v.cols == 1) ? 0 : j;
    return v.mat->get(src_i, src_j);
}

template <typename T>
inline void dense_set_slice(DenseMatrix<T>& mat, const SliceInfo& r, const SliceInfo& c, const AssignmentValue<T>& vinfo) {
    auto [out_r, out_c] = slice_shape(r, c);
    if (!can_broadcast_to(out_r, out_c, vinfo)) {
        throw py::value_error("Right-hand side cannot broadcast to the indexed slice");
    }

    for (uint64_t rr = 0; rr < out_r; ++rr) {
        const uint64_t src_r = r.is_scalar ? r.index : static_cast<uint64_t>(r.start + static_cast<py::ssize_t>(rr) * r.step);
        for (uint64_t cc = 0; cc < out_c; ++cc) {
            const uint64_t src_c = c.is_scalar ? c.index : static_cast<uint64_t>(c.start + static_cast<py::ssize_t>(cc) * c.step);
            mat.set(src_r, src_c, value_at(vinfo, rr, cc));
        }
    }
}

template <typename T>
inline void dense_set_advanced(DenseMatrix<T>& mat, const SliceInfo& r, const SliceInfo& c, const AssignmentValue<T>& vinfo) {
    const bool rows_array = r.is_array;
    const bool cols_array = c.is_array;

    if (rows_array && cols_array) {
        const uint64_t len_r = slice_length(r);
        const uint64_t len_c = slice_length(c);
        uint64_t out_len = 0;
        if (len_r == len_c) {
            out_len = len_r;
        } else if (len_r == 1) {
            out_len = len_c;
        } else if (len_c == 1) {
            out_len = len_r;
        } else {
            throw py::value_error("Row and column index arrays must have the same length or be broadcastable (length 1)");
        }
        if (!can_broadcast_to(1, out_len, vinfo)) {
            throw py::value_error("Right-hand side cannot broadcast to the indexed slice");
        }
        for (uint64_t k = 0; k < out_len; ++k) {
            const uint64_t src_r = r.indices[len_r == 1 ? 0 : k];
            const uint64_t src_c = c.indices[len_c == 1 ? 0 : k];
            mat.set(src_r, src_c, value_at(vinfo, 0, k));
        }
        return;
    }

    if (rows_array) {
        const uint64_t out_r = slice_length(r);
        const uint64_t out_c = slice_length(c);
        if (!can_broadcast_to(out_r, out_c, vinfo)) {
            throw py::value_error("Right-hand side cannot broadcast to the indexed slice");
        }
        for (uint64_t rr = 0; rr < out_r; ++rr) {
            const uint64_t src_r = r.indices[rr];
            for (uint64_t cc = 0; cc < out_c; ++cc) {
                const uint64_t src_c = c.is_scalar ? c.index : static_cast<uint64_t>(c.start + static_cast<py::ssize_t>(cc) * c.step);
                mat.set(src_r, src_c, value_at(vinfo, rr, cc));
            }
        }
        return;
    }

    if (cols_array) {
        const uint64_t out_r = slice_length(r);
        const uint64_t out_c = slice_length(c);
        if (!can_broadcast_to(out_r, out_c, vinfo)) {
            throw py::value_error("Right-hand side cannot broadcast to the indexed slice");
        }
        for (uint64_t rr = 0; rr < out_r; ++rr) {
            const uint64_t src_r = r.is_scalar ? r.index : static_cast<uint64_t>(r.start + static_cast<py::ssize_t>(rr) * r.step);
            for (uint64_t cc = 0; cc < out_c; ++cc) {
                const uint64_t src_c = c.indices[cc];
                mat.set(src_r, src_c, value_at(vinfo, rr, cc));
            }
        }
        return;
    }
}

template <typename T>
inline py::object dense_getitem(const DenseMatrix<T>& mat, const py::object& key) {
    auto parsed = parse_matrix_subscript(key, mat.rows(), mat.cols());
    if (parsed.rows.is_scalar && parsed.cols.is_scalar) {
        return py::cast(mat.get(parsed.rows.index, parsed.cols.index));
    }
    auto sliced = slice_dense_matrix(mat, parsed.rows, parsed.cols);
    // NumPy drops an integer-indexed axis (M[0, :] and M[:, 0] are 1-D), and paired index
    // arrays pick elements (M[[0, 1], [2, 1]] is 1-D): the result is a vector (F-112).
    const bool along_cols = parsed.rows.is_scalar || (parsed.rows.is_array && parsed.cols.is_array);
    if (along_cols || parsed.cols.is_scalar) {
        const uint64_t n = along_cols ? sliced->cols() : sliced->rows();
        auto v = std::make_shared<DenseVector<T>>(n);
        for (uint64_t k = 0; k < n; ++k) v->set(k, along_cols ? sliced->get(0, k) : sliced->get(k, 0));
        // get() reads stored values: the vector keeps the view (B = A * 2: B[0, :] is 2 A[0, :]).
        v->set_scalar(sliced->get_scalar());
        v->set_conjugated(sliced->is_conjugated());
        return py::cast(v);
    }
    return py::cast(sliced);
}

template <typename T>
inline void dense_setitem(DenseMatrix<T>& mat, const py::object& key, const py::object& value) {
    auto parsed = parse_matrix_subscript(key, mat.rows(), mat.cols());
    if (parsed.rows.is_scalar && parsed.cols.is_scalar) {
        mat.set(parsed.rows.index, parsed.cols.index, cast_scalar_to<T>(value));
        return;
    }
    const AssignmentValue<T> vinfo = normalize_assignment_value<T>(value);
    if (parsed.rows.is_array || parsed.cols.is_array) {
        dense_set_advanced(mat, parsed.rows, parsed.cols, vinfo);
    } else {
        dense_set_slice(mat, parsed.rows, parsed.cols, vinfo);
    }
}


// ---------------------------------------------------------------------------------------
// Generic (all dtypes, all structures) element access for the structure classes.
// ---------------------------------------------------------------------------------------
inline std::string describe(const MatrixBase& mat) {
    return std::string(pycauset::structure_name(mat.get_structure())) + " matrix of dtype " +
           pycauset::dtype_name(mat.get_data_type());
}

using pycauset::is_complex_dtype;
using pycauset::is_real_float_dtype;

template <typename M>
using element_of_t = std::decay_t<decltype(std::declval<const M&>().get(uint64_t{0}, uint64_t{0}))>;

// Python value -> element type, NumPy assignment semantics: integers must fit (OverflowError
// otherwise, as NumPy 2); floats assigned to integer dtypes truncate toward zero; complex
// values can't go into real dtypes.
template <typename M>
element_of_t<M> element_from_py(const py::object& value) {
    return pycauset_bindings::py_to_element<element_of_t<M>>(value);
}

// Reads one element as the matching Python scalar. Floats and complex go through the
// view-aware readers (scale factor, conjugation); integers and bits are read exactly.
inline py::object element_get(const MatrixBase& mat, uint64_t i, uint64_t j) {
    const DataType dt = mat.get_data_type();
    if (is_complex_dtype(dt)) return py::cast(mat.get_element_as_complex(i, j));
    if (is_real_float_dtype(dt)) return py::float_(mat.get_element_as_double(i, j));
    const bool scaled = mat.get_scalar() != std::complex<double>(1.0, 0.0);
    if (scaled && !pycauset::scalar::is_exact_int_factor(mat.get_scalar())) {
        throw std::logic_error("integer matrix carries a non-integer scale factor");
    }
    const int64_t k = static_cast<int64_t>(mat.get_scalar().real());
    return pycauset::visit_matrix(mat, [&](const auto& typed) -> py::object {
        using E = element_of_t<std::decay_t<decltype(typed)>>;
        const E v = scaled ? pycauset::scalar::wrap_scale(typed.get(i, j), k) : typed.get(i, j);
        if constexpr (std::is_same_v<E, bool>) return py::bool_(v);
        else if constexpr (std::is_integral_v<E>) return py::int_(v);
        else throw std::logic_error("unreachable: non-integer dtype in the exact read path");
    });
}

// One element's value through the view (the one view rule, ViewValue.hpp), in the element
// type: exact wrapping for integer factors.
template <typename M>
auto viewed_element(const M& typed, uint64_t i, uint64_t j, std::complex<double> s, bool conj) {
    using E = element_of_t<M>;
    if constexpr (std::is_same_v<M, pycauset::ComplexFloat16Matrix>) {
        return pycauset::view_value(typed.get(i, j), s, conj);  // its get / set speak complex128
    } else if constexpr (std::is_integral_v<E> && !std::is_same_v<E, bool>) {
        return pycauset::scalar::wrap_scale(typed.get(i, j), static_cast<int64_t>(s.real()));
    } else if constexpr (std::is_same_v<E, std::complex<double>> || std::is_same_v<E, std::complex<float>>) {
        return static_cast<E>(pycauset::view_value(typed.get(i, j), s, conj));
    } else {
        return static_cast<E>(pycauset::view_value(typed.get(i, j), s, false).real());
    }
}

// Before writing into a view that carries a scale factor or conjugation, fold the view into
// private storage so a write reads back exactly what was written (F-045): every stored
// element is rewritten with its viewed value and the view is cleared. Conjugation of real
// values is a no-op flag and is just cleared (writing to `v.conj()` of an int / bool object
// raised, review 3.2 C-13).
inline void fold_value_view_for_write(MatrixBase& mat) {
    const std::complex<double> s = mat.get_scalar();
    const bool conj = mat.is_conjugated() && is_complex_dtype(mat.get_data_type());
    if (s == std::complex<double>(1.0, 0.0) && !conj) {
        mat.set_conjugated(false);
        return;
    }
    using S = pycauset::Structure;
    const auto st = mat.get_structure();
    if (st == S::IDENTITY) throw py::type_error(describe(mat) + " is read-only");
    // Which logical positions are stored (each once). Dense: all of them.
    bool lower = false, diag_too = false;
    if (st == S::TRIANGULAR) {
        lower = mat.is_transposed();
        diag_too = dynamic_cast<const TriangularMatrixBase&>(mat).has_diagonal();
    }
    auto stored = [&](uint64_t i, uint64_t j) {
        switch (st) {
            case S::DENSE: return true;
            case S::DIAGONAL: return i == j;
            case S::SYMMETRIC:
            case S::HERMITIAN:
            case S::SKEW_HERMITIAN: return j >= i;  // the Hermitian kinds were missing (C-7)
            case S::ANTISYMMETRIC: return j > i;
            case S::TRIANGULAR: return lower ? (i > j || (diag_too && i == j)) : (j > i || (diag_too && i == j));
            default: return false;
        }
    };
    pycauset::visit_matrix(mat, [&](auto& typed) {
        using M = std::decay_t<decltype(typed)>;
        using E = element_of_t<M>;
        if constexpr (std::is_same_v<E, bool>) {
            throw std::logic_error("bit matrix carries a scale factor");
        } else if constexpr (requires { typed.set(uint64_t{0}, uint64_t{0}, viewed_element(typed, 0, 0, s, conj)); }) {
            using V = decltype(viewed_element(typed, 0, 0, s, conj));
            std::vector<std::tuple<uint64_t, uint64_t, V>> values;
            const uint64_t r = typed.rows(), c = typed.cols();
            for (uint64_t i = 0; i < r; ++i)
                for (uint64_t j = 0; j < c; ++j)
                    if (stored(i, j)) values.emplace_back(i, j, viewed_element(typed, i, j, s, conj));
            mat.set_scalar(1.0);
            mat.set_conjugated(false);
            for (const auto& [i, j, v] : values) typed.set(i, j, v);
        } else {
            throw py::type_error(describe(mat) + " cannot be written through a view");
        }
    });
    mat.set_scalar(1.0);
    mat.set_conjugated(false);
}

inline void element_set(MatrixBase& mat, uint64_t i, uint64_t j, const py::object& value) {
    fold_value_view_for_write(mat);
    pycauset::visit_matrix(mat, [&](auto& typed) {
        using M = std::decay_t<decltype(typed)>;
        if constexpr (requires { typed.set(i, j, element_from_py<M>(value)); }) {
            typed.set(i, j, element_from_py<M>(value));
        } else {
            throw py::type_error(describe(mat) + " is read-only");
        }
    });
}

// Slice / fancy-index assignment for structures without a bulk path (triangular, symmetric,
// diagonal, complex16): element by element with NumPy semantics. Two index arrays pair up
// (A[[0, 1], [2, 3]] = v sets (0,2) and (1,3)); otherwise rows x cols. The value is a scalar
// or has one entry per target element.
inline std::vector<uint64_t> slice_positions(const SliceInfo& s) {
    if (s.is_scalar) return {s.index};
    if (s.is_array) return s.indices;
    std::vector<uint64_t> out;
    out.reserve(s.length);
    for (uint64_t k = 0; k < s.length; ++k) out.push_back(static_cast<uint64_t>(s.start + static_cast<py::ssize_t>(k) * s.step));
    return out;
}

inline void element_set(MatrixBase& mat, uint64_t i, uint64_t j, const py::object& value);

inline void setitem_elementwise(MatrixBase& mat, const ParsedIndex& parsed, const py::object& value) {
    const auto rows = slice_positions(parsed.rows);
    const auto cols = slice_positions(parsed.cols);
    std::vector<std::pair<uint64_t, uint64_t>> targets;
    if (parsed.rows.is_array && parsed.cols.is_array) {
        if (rows.size() != cols.size()) throw py::value_error("row and column index arrays must have the same length");
        for (size_t k = 0; k < rows.size(); ++k) targets.emplace_back(rows[k], cols[k]);
    } else {
        for (uint64_t i : rows) for (uint64_t j : cols) targets.emplace_back(i, j);
    }
    py::array values = py::array::ensure(value);
    if (!values) throw py::type_error("assigned value must be a scalar or array-like");
    py::list flat = values.attr("ravel")().attr("tolist")();
    const size_t nvals = py::len(flat);
    if (nvals != 1 && nvals != targets.size()) {
        throw py::value_error("cannot broadcast " + std::to_string(nvals) + " values to " + std::to_string(targets.size()) + " elements");
    }
    for (size_t k = 0; k < targets.size(); ++k) {
        element_set(mat, targets[k].first, targets[k].second, flat[nvals == 1 ? 0 : k]);
    }
}

inline py::object matrixbase_getitem_dispatch(const MatrixBase& mat, const py::object& key) {
    auto parsed = parse_matrix_subscript(key, mat.rows(), mat.cols());
    if (parsed.rows.is_scalar && parsed.cols.is_scalar) {
        return element_get(mat, parsed.rows.index, parsed.cols.index);
    }
    return pycauset::visit_matrix(mat, [&](const auto& typed) -> py::object {
        if constexpr (requires { dense_getitem(typed, key); }) {
            return dense_getitem(typed, key);
        } else {
            // A row or column (an integer index on one axis, a slice on the other) of any
            // structure: a vector of those elements, as NumPy gives (e.g. the past of an
            // element is a column of the causal matrix: O(n), not an O(n^2) export). Read
            // with the element accessor, so every view rule applies.
            const SliceInfo& r = parsed.rows;
            const SliceInfo& c = parsed.cols;
            if ((r.is_scalar != c.is_scalar) && !r.is_array && !c.is_array) {
                const SliceInfo& s = r.is_scalar ? c : r;
                py::list items;
                for (uint64_t k = 0; k < s.length; ++k) {
                    const uint64_t t = static_cast<uint64_t>(s.start + static_cast<py::ssize_t>(k) * s.step);
                    items.append(r.is_scalar ? element_get(mat, r.index, t) : element_get(mat, t, c.index));
                }
                return py::module_::import("pycauset").attr("vector")(
                    items, py::arg("dtype") = pycauset::dtype_name(mat.get_data_type()));
            }
            PyErr_SetString(PyExc_NotImplementedError, ("slicing is not supported for " + describe(mat)).c_str());
            throw py::error_already_set();
        }
    });
}

inline void matrixbase_setitem_dispatch(MatrixBase& mat, const py::object& key, const py::object& value) {
    auto parsed = parse_matrix_subscript(key, mat.rows(), mat.cols());
    if (parsed.rows.is_scalar && parsed.cols.is_scalar) {
        element_set(mat, parsed.rows.index, parsed.cols.index, value);
        return;
    }
    fold_value_view_for_write(mat);
    pycauset::visit_matrix(mat, [&](auto& typed) {
        if constexpr (requires { dense_setitem(typed, key, value); }) {
            dense_setitem(typed, key, value);
        } else {
            setitem_elementwise(mat, parsed, value);
        }
    });
}

}  // namespace pycauset_bindings::indexing
