// NumPy -> engine import helpers and binding warnings shared by bind_matrix.cpp and
// bind_vector.cpp (review 4.2: the two files had diverging copies). Internal to src/bindings.
#pragma once

#include <bit>

#include "bindings_common.hpp"
#include "element_conversion.hpp"
#include "binding_warnings.hpp"
#include "numpy_dtypes.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/math/VectorOps.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/VectorBase.hpp"

#include "pycauset/core/ObjectFactory.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <stdexcept>
#include <string>

namespace pycauset_bindings::numpy_import {
using namespace pycauset;

// One normalisation of NumPy input for every importer: native byte order, aligned, and strides
// that are whole items. A big-endian float16 imported byte-swapped, a struct-field view with a
// 9-byte stride read the wrong memory, and '>f8' / '>i4' raised (review 3.7 C-3). Arrays
// already in that form are returned as they are (no copy).
inline py::array normalize_numpy(const py::array& a) {
    const py::object dt = a.attr("dtype");
    const std::string bo = dt.attr("byteorder").cast<std::string>();
    const bool swapped = (std::endian::native == std::endian::little) ? bo == ">" : bo == "<";
    bool odd = false;
    const py::ssize_t item = a.itemsize();
    for (py::ssize_t d = 0; d < a.ndim(); ++d) {
        if (item > 0 && a.strides(d) % item != 0) odd = true;
    }
    const bool aligned = a.attr("flags").attr("aligned").cast<bool>();
    if (!swapped && !odd && aligned) return a;
    py::module_ np = py::module_::import("numpy");
    return np.attr("ascontiguousarray")(a, dt.attr("newbyteorder")("=")).cast<py::array>();
}

// Bits of float precision of a float / complex dtype (0 for the others).
inline int float_precision_bits(DataType dt) {
    switch (dt) {
        case DataType::FLOAT16: case DataType::COMPLEX_FLOAT16: return 16;
        case DataType::FLOAT32: case DataType::COMPLEX_FLOAT32: return 32;
        case DataType::FLOAT64: case DataType::COMPLEX_FLOAT64: return 64;
        default: return 0;
    }
}

// A float or complex result narrower than one of its float / complex operands differs from
// NumPy (which widens): say so once per call site. It fired for float32 with float64 only;
// float16 and complex mixes were silent (review 3.3 C-6). Every operand kind comes here
// (matrices, vectors, NumPy scalars, matrix-vector products), either order: the warning
// depended on the binding and on operand order (review 3.7 C-10).
inline void maybe_warn_float_underpromotion(const char* op_name, promotion::BinaryOp op, DataType da, DataType db) {
    const int pa = float_precision_bits(da), pb = float_precision_bits(db);
    if (pa == 0 || pb == 0 || pa == pb) return;  // only a mix of float / complex widths
    const DataType r = promotion::resolve(op, da, db).result_dtype;
    if (float_precision_bits(r) >= std::max(pa, pb)) return;

    std::string message = std::string("pycauset ") + op_name + ": " + pycauset::dtype_name(da) + " with " + pycauset::dtype_name(db) +
                          " gives " + pycauset::dtype_name(r) + " (computed and stored in it; NumPy would widen). " +
                          "Cast the narrower operand first to compute in the wider dtype.";
    bindings_warn::warn_with_category(message, "PyCausetDTypeWarning", /*stacklevel=*/3);
}

inline void maybe_warn_float_underpromotion(const char* op_name, promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b) {
    maybe_warn_float_underpromotion(op_name, op, a.get_data_type(), b.get_data_type());
}

inline void maybe_warn_bit_promotes_to_int32(const char* op_name, promotion::BinaryOp op, DataType da, DataType db) {
    if (!(da == DataType::BIT && db == DataType::BIT)) {
        return;
    }

    const auto decision = promotion::resolve(op, da, db);
    if (decision.result_dtype != DataType::INT32) {
        return;
    }

    std::string message;
    message.reserve(256);
    message += "pycauset ";
    message += op_name;
    message += ": bit operands promote to int32 because the result can exceed {0,1} (mathematically unavoidable). ";
    message += "Even though int16 exists, bit-bit reductions commonly exceed int16; result is int32.";

    bindings_warn::warn_with_category(message, "PyCausetDTypeWarning", /*stacklevel=*/3);
}

// --- NumPy -> engine: one importer (review 3.7 C-12) -------------------------------------
// Every path (pc.asarray, operands of arithmetic, comparisons, bitwise and matmul, vectors,
// rows) comes here: one normalisation (normalize_numpy), one dtype table, one templated
// strided copy and one bool packer. There were five per-dtype ladders that had diverged (the
// operand path set bool bits one call at a time: 40 ms for a 2000^2 operand, 1 ms through
// pc.asarray).

// rows x cols elements at src (strides s0, s1 in items) into the row-major dst. Parallel, with
// the GIL released, from 1 MB.
template <typename E>
void copy_strided(const E* src, ptrdiff_t s0, ptrdiff_t s1, uint64_t rows, uint64_t cols, E* dst) {
    if (rows == 0 || cols == 0) return;
    const bool parallel = rows * cols * sizeof(E) >= (size_t{1} << 20);
    std::optional<py::gil_scoped_release> release;
    if (parallel) release.emplace();
    if (s1 == 1 && (rows == 1 || s0 == static_cast<ptrdiff_t>(cols))) {  // contiguous
        if (!parallel) { std::memcpy(dst, src, rows * cols * sizeof(E)); return; }
        pycauset::ParallelFor(0, rows, [&](size_t i) { std::memcpy(dst + i * cols, src + i * cols, cols * sizeof(E)); });
        return;
    }
    if (s1 == 1) {  // rows contiguous, padding between them
        auto row = [&](size_t i) { std::memcpy(dst + i * cols, src + static_cast<ptrdiff_t>(i) * s0, cols * sizeof(E)); };
        if (parallel) pycauset::ParallelFor(0, rows, row); else for (uint64_t i = 0; i < rows; ++i) row(i);
        return;
    }
    if (rows > 1 && (s0 == 1 || s0 == -1)) {  // columns contiguous (a transpose): 32 x 32 tiles
        constexpr uint64_t B = 32;
        auto tiles = [&](size_t jt) {
            alignas(64) E tile[B * B];
            const uint64_t j0 = jt * B, jb = std::min<uint64_t>(B, cols - j0);
            for (uint64_t i0 = 0; i0 < rows; i0 += B) {
                const uint64_t ib = std::min<uint64_t>(B, rows - i0);
                for (uint64_t j = 0; j < jb; ++j) {
                    const E* col = src + static_cast<ptrdiff_t>(i0) * s0 + static_cast<ptrdiff_t>(j0 + j) * s1;
                    for (uint64_t i = 0; i < ib; ++i) tile[i * B + j] = col[static_cast<ptrdiff_t>(i) * s0];
                }
                for (uint64_t i = 0; i < ib; ++i) std::memcpy(dst + (i0 + i) * cols + j0, tile + i * B, jb * sizeof(E));
            }
        };
        const uint64_t nt = (cols + B - 1) / B;
        if (parallel) pycauset::ParallelFor(0, nt, tiles); else for (uint64_t t = 0; t < nt; ++t) tiles(t);
        return;
    }
    auto row = [&](size_t i) {  // general strides
        const E* p = src + static_cast<ptrdiff_t>(i) * s0;
        for (uint64_t j = 0; j < cols; ++j) dst[i * cols + j] = p[static_cast<ptrdiff_t>(j) * s1];
    };
    if (parallel) pycauset::ParallelFor(0, rows, row); else for (uint64_t i = 0; i < rows; ++i) row(i);
}

// NumPy bools (one byte each, 0 or 1) into packed rows of 64-bit words (stride in words).
inline void pack_bools(const bool* src, ptrdiff_t s0, ptrdiff_t s1, uint64_t rows, uint64_t cols, uint64_t* words,
                       uint64_t stride_words) {
    auto row = [&](size_t i) {
        const bool* r = src + static_cast<ptrdiff_t>(i) * s0;
        uint64_t* out = words + i * stride_words;
        for (uint64_t j = 0; j < cols; j += 64) {
            const uint64_t limit = std::min<uint64_t>(cols - j, 64);
            uint64_t word = 0, k = 0;
            if (s1 == 1) {
                const bool* chunk = r + j;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                const __m128i zero = _mm_setzero_si128();
                for (; k + 16 <= limit; k += 16) {
                    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(chunk + k));
                    word |= static_cast<uint64_t>(static_cast<uint16_t>(~_mm_movemask_epi8(_mm_cmpeq_epi8(v, zero)))) << k;
                }
#endif
                for (; k < limit; ++k) word |= static_cast<uint64_t>(chunk[k] != 0) << k;
            } else {
                for (; k < limit; ++k) word |= static_cast<uint64_t>(r[static_cast<ptrdiff_t>(j + k) * s1] != 0) << k;
            }
            out[j / 64] = word;
        }
    };
    if (rows * cols >= (uint64_t{1} << 20)) {
        py::gil_scoped_release release;
        pycauset::ParallelFor(0, rows, row);
    } else {
        for (uint64_t i = 0; i < rows; ++i) row(i);
    }
}

// Copies a normalised array's elements into `out` (a dense matrix or vector of the array's dtype).
template <typename Obj>
void fill_from_numpy(const py::buffer_info& buf, ptrdiff_t s0, ptrdiff_t s1, uint64_t rows, uint64_t cols, DataType dt, Obj& out) {
    pycauset::visit_dtype(dt, [&](auto tag) {
        using T = typename decltype(tag)::type;
        using Target = std::conditional_t<std::is_base_of_v<MatrixBase, Obj>, typename pycauset::DenseMatrixOf<T>::type,
                                          typename pycauset::DenseVectorOf<T>::type>;
        if constexpr (std::is_same_v<T, pycauset::complex_float16_marker>) {
            throw std::logic_error("NumPy has no complex_float16");
        } else {
            auto& o = dynamic_cast<Target&>(out);
            if constexpr (std::is_same_v<T, bool>) {
                uint64_t stride_words = (cols + 63) / 64;
                if constexpr (std::is_base_of_v<MatrixBase, Obj>) stride_words = o.stride_bytes() / 8;
                pack_bools(static_cast<const bool*>(buf.ptr), s0, s1, rows, cols, o.data(), stride_words);
            } else if constexpr (std::is_same_v<T, pycauset::float16_t>) {  // NumPy float16: the bits
                copy_strided(static_cast<const uint16_t*>(buf.ptr), s0, s1, rows, cols, reinterpret_cast<uint16_t*>(o.data()));
            } else {
                copy_strided(static_cast<const T*>(buf.ptr), s0, s1, rows, cols, o.data());
            }
        }
    });
}

enum class ImportAs { Vector, Matrix, Row };  // Row: a 1-D array as a 1 x n matrix

struct NumpyLayout {
    py::array array;  // normalised (kept alive while the copy reads it)
    py::buffer_info buf;
    DataType dt;
    uint64_t rows, cols;
    ptrdiff_t s0, s1;
};

inline NumpyLayout numpy_layout(const py::array& array_in, ImportAs as) {
    NumpyLayout l{normalize_numpy(array_in), {}, DataType::UNKNOWN, 0, 0, 0, 0};
    l.buf = l.array.request();
    const int want = as == ImportAs::Matrix ? 2 : 1;
    if (l.buf.ndim != want) {
        throw py::value_error(want == 2 ? "Number of dimensions must be 2" : "Expected a 1D array");
    }
    l.dt = pycauset_bindings::dtype_from_numpy(l.array.dtype());
    if (l.dt == DataType::UNKNOWN) {
        throw py::type_error("Unsupported NumPy dtype " + py::str(l.array.dtype()).cast<std::string>() + " for a pycauset object");
    }
    const auto item = static_cast<py::ssize_t>(l.buf.itemsize);
    l.rows = want == 2 ? static_cast<uint64_t>(l.buf.shape[0]) : 1;
    l.cols = static_cast<uint64_t>(l.buf.shape[want - 1]);
    l.s0 = want == 2 ? static_cast<ptrdiff_t>(l.buf.strides[0] / item) : static_cast<ptrdiff_t>(l.cols);
    l.s1 = static_cast<ptrdiff_t>(l.buf.strides[want - 1] / item);
    return l;
}

inline std::shared_ptr<MatrixBase> import_numpy_matrix(const py::array& array, ImportAs as) {
    const NumpyLayout l = numpy_layout(array, as);
    // No RAM check: allocation puts an object above the memory threshold (or one that does not
    // fit in RAM) in a file.
    std::shared_ptr<MatrixBase> out = pycauset::ObjectFactory::create_matrix(l.rows, l.cols, l.dt, pycauset::Structure::DENSE);
    fill_from_numpy(l.buf, l.s0, l.s1, l.rows, l.cols, l.dt, *out);
    return out;
}

inline std::shared_ptr<VectorBase> vector_from_numpy(const py::array& array) {
    const NumpyLayout l = numpy_layout(array, ImportAs::Vector);
    std::shared_ptr<VectorBase> out = pycauset::ObjectFactory::create_vector(l.cols, l.dt, pycauset::Structure::VECTOR);
    fill_from_numpy(l.buf, l.s0, l.s1, l.rows, l.cols, l.dt, *out);
    return out;
}

inline std::shared_ptr<MatrixBase> matrix_from_numpy(const py::array& array) {
    return import_numpy_matrix(array, ImportAs::Matrix);
}

// An operand of an element-wise operation: 2-D as is, 1-D as a row (NumPy broadcasting).
inline std::shared_ptr<MatrixBase> elementwise_operand_matrix_from_numpy(const py::array& array) {
    const auto nd = array.ndim();
    if (nd != 1 && nd != 2) throw py::value_error("Unsupported array rank for elementwise operation");
    return import_numpy_matrix(array, nd == 2 ? ImportAs::Matrix : ImportAs::Row);
}

template <typename Fn>
inline auto translate_invalid_argument(Fn&& fn) {
    try {
        return fn();
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
}

// The other operand of a comparison as an engine matrix (nullptr: not an operand, so the
// operator returns NotImplemented): a matrix as is, a vector as its 1 x n row, an ndarray
// through the one normalised import (1-D: a row, as in arithmetic), a Python or NumPy scalar as a
// 1 x 1 matrix of its NumPy dtype.
inline std::shared_ptr<MatrixBase> compare_operand(const py::object& other) {
    if (py::isinstance<MatrixBase>(other)) return other.cast<std::shared_ptr<MatrixBase>>();
    if (py::isinstance<VectorBase>(other)) {
        return std::shared_ptr<MatrixBase>(pycauset::as_row_matrix(other.cast<const VectorBase&>()).release());
    }
    if (py::isinstance<py::array>(other)) return elementwise_operand_matrix_from_numpy(other.cast<py::array>());
    PyObject* p = other.ptr();
    py::module_ np = py::module_::import("numpy");
    if (PyBool_Check(p) || PyLong_Check(p) || PyFloat_Check(p) || PyComplex_Check(p) ||
        py::isinstance(other, np.attr("generic"))) {
        py::array arr = np.attr("asarray")(other).attr("reshape")(1, 1);
        if (arr.dtype().kind() == 'O') return nullptr;  // a Python int beyond 64 bits
        return matrix_from_numpy(arr);
    }
    return nullptr;
}

}  // namespace pycauset_bindings::numpy_import
