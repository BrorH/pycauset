// NumPy -> engine import helpers and binding warnings shared by bind_matrix.cpp and
// bind_vector.cpp (review 4.2: the two files had diverging copies). Internal to src/bindings.
#pragma once

#include "bindings_common.hpp"
#include "element_conversion.hpp"
#include "binding_warnings.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
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

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace pycauset_bindings::numpy_import {
using namespace pycauset;

inline bool is_numpy_float16(const py::array& array) {
    const py::dtype dt = array.dtype();
    return dt.kind() == 'f' && dt.itemsize() == 2;
}

inline void maybe_warn_float_underpromotion(const char* op_name, promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b) {
    DataType da = a.get_data_type();
    DataType db = b.get_data_type();

    bool mixed = (da == DataType::FLOAT32 && db == DataType::FLOAT64) ||
                 (da == DataType::FLOAT64 && db == DataType::FLOAT32);
    if (!mixed) {
        return;
    }

    const auto decision = promotion::resolve(op, da, db);
    if (!decision.float_underpromotion) {
        return;
    }

    std::string message;
    message.reserve(256);
    message += "pycauset ";
    message += op_name;
    message += ": mixed float32/float64 underpromotes to float32 (compute and storage in float32). ";
    message += "Cast both inputs to float64 to compute in float64.";

    bindings_warn::warn_with_category(message, "PyCausetDTypeWarning", /*stacklevel=*/3);
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

inline void require_2d(const py::buffer_info& buf) {
    if (buf.ndim != 2) {
        throw py::value_error("Number of dimensions must be 2");
    }
}

inline void require_1d(const py::buffer_info& buf) {
    if (buf.ndim != 1) {
        throw py::value_error("Expected a 1D array");
    }
}

template <typename T>
std::shared_ptr<DenseMatrix<T>> dense_matrix_from_numpy_2d(const py::array_t<T>& array) {
    auto buf = array.request();
    require_2d(buf);

    uint64_t rows = static_cast<uint64_t>(buf.shape[0]);
    uint64_t cols = static_cast<uint64_t>(buf.shape[1]);

    // No RAM check here: storage allocation already puts an object above the memory threshold
    // (or one that does not fit in RAM) in a file. A RAM check refused large arrays that would
    // have been file-backed anyway.
    auto result = std::make_shared<DenseMatrix<T>>(rows, cols);

    const T* src_ptr = static_cast<const T*>(buf.ptr);
    T* dst_ptr = result->data();

    const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
    const ptrdiff_t stride1 = static_cast<ptrdiff_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(T)));

    // PHASE 3.5: Advanced Strided Optimizations (Parallel Import)
    // We release the GIL to allow multi-threaded memory copy if the matrix is large enough.
    // Heuristic: If total bytes > 1MB, use parallel path.
    constexpr size_t PARALLEL_THRESHOLD_BYTES = 1024 * 1024;
    size_t total_bytes = rows * cols * sizeof(T);
    
    // Check if we can use the parallel path (requires GIL release)
    if (total_bytes >= PARALLEL_THRESHOLD_BYTES) {
        py::gil_scoped_release release;
        
        // Fast path: fully C-contiguous.
        if (stride1 == 1 && stride0 == static_cast<ptrdiff_t>(cols)) {
            // Parallel memcpy for huge contiguous arrays
            // Split by rows or chunks.
            pycauset::ParallelFor(0, rows, [&](size_t i) {
                const T* src_row = src_ptr + i * cols;
                T* dst_row = dst_ptr + i * cols;
                std::memcpy(dst_row, src_row, cols * sizeof(T));
            });
            return result;
        }

        // Fast path: row-contiguous (possibly with padding between rows).
        if (stride1 == 1) {
            pycauset::ParallelFor(0, rows, [&](size_t i) {
                const T* src_row = src_ptr + static_cast<ptrdiff_t>(i) * stride0;
                T* dst_row = dst_ptr + i * cols;
                std::memcpy(dst_row, src_row, cols * sizeof(T));
            });
            return result;
        }

        // General path: Strided copy
        // Parallelize over rows.
        pycauset::ParallelFor(0, rows, [&](size_t i) {
            const T* src_row = src_ptr + static_cast<ptrdiff_t>(i) * stride0;
            T* dst_row = dst_ptr + i * cols;

            // Inner loop: Check if we have stride optimization opportunities
            if (stride1 == -1) {
               // Reversed column copy optimization
               const T* p = src_row;
               for (uint64_t j = 0; j < cols; ++j) {
                   dst_row[j] = *p;
                   p += stride1;
               }
            } else {
               // Generic heavy loop
               const T* p = src_row;
               for (uint64_t j = 0; j < cols; ++j) {
                   dst_row[j] = *p;
                   p += stride1; // Pointer arithmetic for next element
               }
            }
        });

        return result;
    }

    // --- Fallback to single-threaded path for small arrays (keeping GIL is fine/faster) ---

    // Fast path: fully C-contiguous.
    if (stride1 == 1 && stride0 == static_cast<ptrdiff_t>(cols)) {
        std::memcpy(dst_ptr, src_ptr, rows * cols * sizeof(T));
        return result;
    }

    // Fast path: row-contiguous (possibly with padding between rows).
    if (stride1 == 1) {
        for (uint64_t i = 0; i < rows; ++i) {
            const T* src_row = src_ptr + static_cast<ptrdiff_t>(i) * stride0;
            T* dst_row = dst_ptr + i * cols;
            std::memcpy(dst_row, src_row, cols * sizeof(T));
        }
        return result;
    }

    // Fast path: row-contiguous but reversed along columns (e.g., arr[:, ::-1]).
    if (stride1 == -1) {
        for (uint64_t i = 0; i < rows; ++i) {
            const T* src_row = src_ptr + static_cast<ptrdiff_t>(i) * stride0;
            T* dst_row = dst_ptr + i * cols;
            const T* p = src_row;
            for (uint64_t j = 0; j < cols; ++j) {
                dst_row[j] = *p;
                p += stride1;
            }
        }
        return result;
    }

    // Fast path: contiguous along the first dimension (common for transpose / Fortran-like layouts).
    // Use a small tile buffer to achieve both contiguous reads and contiguous writes.
    if (stride0 == 1 || stride0 == -1) {
        constexpr uint64_t B = 32;
        alignas(64) T tile[B * B];

        for (uint64_t j0 = 0; j0 < cols; j0 += B) {
            const uint64_t jb = std::min<uint64_t>(B, cols - j0);
            for (uint64_t i0 = 0; i0 < rows; i0 += B) {
                const uint64_t ib = std::min<uint64_t>(B, rows - i0);

                for (uint64_t j = 0; j < jb; ++j) {
                    const T* src_col = src_ptr + static_cast<ptrdiff_t>(i0) * stride0 +
                                       static_cast<ptrdiff_t>(j0 + j) * stride1;
                    for (uint64_t i = 0; i < ib; ++i) {
                        tile[i * B + j] = src_col[static_cast<ptrdiff_t>(i) * stride0];
                    }
                }

                for (uint64_t i = 0; i < ib; ++i) {
                    T* dst_row = dst_ptr + (i0 + i) * cols + j0;
                    std::memcpy(dst_row, tile + i * B, jb * sizeof(T));
                }
            }
        }
        return result;
    }

    // Generic strided fallback.
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0 + static_cast<ptrdiff_t>(j) * stride1;
            dst_ptr[i * cols + j] = src_ptr[idx];
        }
    }
    return result;
}

template <typename T>
std::shared_ptr<DenseVector<T>> dense_vector_from_numpy_1d(const py::array_t<T>& array) {
    auto buf = array.request();
    require_1d(buf);
    uint64_t n = static_cast<uint64_t>(buf.shape[0]);
    auto result = std::make_shared<DenseVector<T>>(n);

    const T* src_ptr = static_cast<const T*>(buf.ptr);
    T* dst_ptr = result->data();
    const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
    if (stride0 == 1) {
        std::memcpy(dst_ptr, src_ptr, n * sizeof(T));
    } else {
        for (uint64_t i = 0; i < n; ++i) {
            const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0;
            dst_ptr[i] = src_ptr[idx];
        }
    }
    return result;
}


template <typename T>
std::shared_ptr<DenseMatrix<T>> dense_row_matrix_from_numpy_1d(const py::array_t<T>& array) {
    auto buf = array.request();
    require_1d(buf);

    uint64_t cols = static_cast<uint64_t>(buf.shape[0]);
    auto result = std::make_shared<DenseMatrix<T>>(1, cols);

    const T* src_ptr = static_cast<const T*>(buf.ptr);
    T* dst_ptr = result->data();
    const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
    if (stride0 == 1) {
        std::memcpy(dst_ptr, src_ptr, cols * sizeof(T));
    } else {
        for (uint64_t j = 0; j < cols; ++j) {
            const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * stride0;
            dst_ptr[j] = src_ptr[idx];
        }
    }
    return result;
}

inline std::shared_ptr<VectorBase> vector_from_numpy(const py::array& array) {
    if (is_numpy_float16(array)) {  // NumPy float16 has no C++ type in pybind11: read the bits
        auto buf = array.request();
        require_1d(buf);
        const uint64_t n = static_cast<uint64_t>(buf.shape[0]);
        auto result = std::make_shared<DenseVector<pycauset::float16_t>>(n);
        const auto* src = static_cast<const uint16_t*>(buf.ptr);
        const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
        pycauset::float16_t* dst = result->data();
        for (uint64_t i = 0; i < n; ++i) {
            dst[i] = pycauset::float16_t(src[static_cast<ptrdiff_t>(i) * stride0]);
        }
        return result;
    }
    if (py::isinstance<py::array_t<double>>(array)) {
        return dense_vector_from_numpy_1d<double>(array.cast<py::array_t<double>>());
    }
    if (py::isinstance<py::array_t<int32_t>>(array)) {
        return dense_vector_from_numpy_1d<int32_t>(array.cast<py::array_t<int32_t>>());
    }
    if (py::isinstance<py::array_t<int64_t>>(array)) {
        return dense_vector_from_numpy_1d<int64_t>(array.cast<py::array_t<int64_t>>());
    }
    if (py::isinstance<py::array_t<int8_t>>(array)) {
        return dense_vector_from_numpy_1d<int8_t>(array.cast<py::array_t<int8_t>>());
    }
    if (py::isinstance<py::array_t<int16_t>>(array)) {
        return dense_vector_from_numpy_1d<int16_t>(array.cast<py::array_t<int16_t>>());
    }
    if (py::isinstance<py::array_t<uint8_t>>(array)) {
        return dense_vector_from_numpy_1d<uint8_t>(array.cast<py::array_t<uint8_t>>());
    }
    if (py::isinstance<py::array_t<uint16_t>>(array)) {
        return dense_vector_from_numpy_1d<uint16_t>(array.cast<py::array_t<uint16_t>>());
    }
    if (py::isinstance<py::array_t<uint32_t>>(array)) {
        return dense_vector_from_numpy_1d<uint32_t>(array.cast<py::array_t<uint32_t>>());
    }
    if (py::isinstance<py::array_t<uint64_t>>(array)) {
        return dense_vector_from_numpy_1d<uint64_t>(array.cast<py::array_t<uint64_t>>());
    }
    if (py::isinstance<py::array_t<bool>>(array)) {
        auto tmp = array.cast<py::array_t<bool>>();
        auto buf = tmp.request();
        require_1d(buf);
        uint64_t n = static_cast<uint64_t>(buf.shape[0]);
        auto result = std::make_shared<DenseVector<bool>>(n);
        const bool* src_ptr = static_cast<const bool*>(buf.ptr);
        const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
        for (uint64_t i = 0; i < n; ++i) {
            const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0;
            result->set(i, src_ptr[idx]);
        }
        return result;
    }
    if (py::isinstance<py::array_t<float>>(array)) {
        return dense_vector_from_numpy_1d<float>(array.cast<py::array_t<float>>());  // dtype kept (F-056)
    }
    if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
        return dense_vector_from_numpy_1d<std::complex<float>>(array.cast<py::array_t<std::complex<float>>>());
    }
    if (py::isinstance<py::array_t<std::complex<double>>>(array)) {
        return dense_vector_from_numpy_1d<std::complex<double>>(array.cast<py::array_t<std::complex<double>>>());
    }
    throw py::type_error("Unsupported NumPy dtype for vector conversion");
}

inline std::shared_ptr<MatrixBase> matrix_from_numpy(const py::array& array) {
    if (is_numpy_float16(array)) {
        auto buf = array.request();
        require_2d(buf);

        uint64_t rows = static_cast<uint64_t>(buf.shape[0]);
        uint64_t cols = static_cast<uint64_t>(buf.shape[1]);
        auto result = std::make_shared<DenseMatrix<pycauset::float16_t>>(rows, cols);

        const uint16_t* src_ptr = static_cast<const uint16_t*>(buf.ptr);
        pycauset::float16_t* dst_ptr = result->data();

        if (buf.strides[1] == static_cast<py::ssize_t>(sizeof(uint16_t)) &&
            buf.strides[0] == static_cast<py::ssize_t>(cols * sizeof(uint16_t))) {
            std::memcpy(static_cast<void*>(dst_ptr), src_ptr, rows * cols * sizeof(uint16_t));
        } else {
            const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
            const ptrdiff_t stride1 = static_cast<ptrdiff_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(uint16_t)));
            for (uint64_t i = 0; i < rows; ++i) {
                for (uint64_t j = 0; j < cols; ++j) {
                    const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0 + static_cast<ptrdiff_t>(j) * stride1;
                    dst_ptr[i * cols + j] = pycauset::float16_t(static_cast<uint16_t>(src_ptr[idx]));
                }
            }
        }
        return result;
    }
    if (py::isinstance<py::array_t<double>>(array)) {
        return dense_matrix_from_numpy_2d<double>(array.cast<py::array_t<double>>());
    }
    if (py::isinstance<py::array_t<float>>(array)) {
        return dense_matrix_from_numpy_2d<float>(array.cast<py::array_t<float>>());
    }
    if (py::isinstance<py::array_t<int32_t>>(array)) {
        return dense_matrix_from_numpy_2d<int32_t>(array.cast<py::array_t<int32_t>>());
    }
    if (py::isinstance<py::array_t<int64_t>>(array)) {
        return dense_matrix_from_numpy_2d<int64_t>(array.cast<py::array_t<int64_t>>());
    }
    if (py::isinstance<py::array_t<int8_t>>(array)) {
        return dense_matrix_from_numpy_2d<int8_t>(array.cast<py::array_t<int8_t>>());
    }
    if (py::isinstance<py::array_t<int16_t>>(array)) {
        return dense_matrix_from_numpy_2d<int16_t>(array.cast<py::array_t<int16_t>>());
    }
    if (py::isinstance<py::array_t<uint8_t>>(array)) {
        return dense_matrix_from_numpy_2d<uint8_t>(array.cast<py::array_t<uint8_t>>());
    }
    if (py::isinstance<py::array_t<uint16_t>>(array)) {
        return dense_matrix_from_numpy_2d<uint16_t>(array.cast<py::array_t<uint16_t>>());
    }
    if (py::isinstance<py::array_t<uint32_t>>(array)) {
        return dense_matrix_from_numpy_2d<uint32_t>(array.cast<py::array_t<uint32_t>>());
    }
    if (py::isinstance<py::array_t<uint64_t>>(array)) {
        return dense_matrix_from_numpy_2d<uint64_t>(array.cast<py::array_t<uint64_t>>());
    }
    if (py::isinstance<py::array_t<bool>>(array)) {
        auto tmp = array.cast<py::array_t<bool>>();
        auto buf = tmp.request();
        require_2d(buf);
        uint64_t rows = static_cast<uint64_t>(buf.shape[0]);
        uint64_t cols = static_cast<uint64_t>(buf.shape[1]);
        auto result = std::make_shared<DenseBitMatrix>(rows, cols);
        const bool* src_ptr = static_cast<const bool*>(buf.ptr);
        const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
        const ptrdiff_t stride1 = static_cast<ptrdiff_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(bool)));
        for (uint64_t i = 0; i < rows; ++i) {
            for (uint64_t j = 0; j < cols; ++j) {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0 + static_cast<ptrdiff_t>(j) * stride1;
                result->set(i, j, src_ptr[idx]);
            }
        }
        return result;
    }
    if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
        return dense_matrix_from_numpy_2d<std::complex<float>>(array.cast<py::array_t<std::complex<float>>>());
    }
    if (py::isinstance<py::array_t<std::complex<double>>>(array)) {
        return dense_matrix_from_numpy_2d<std::complex<double>>(array.cast<py::array_t<std::complex<double>>>());
    }
    throw py::type_error("Unsupported NumPy dtype for matrix conversion");
}

inline std::shared_ptr<MatrixBase> elementwise_operand_matrix_from_numpy(const py::array& array) {
    auto buf = array.request();
    if (buf.ndim == 2) {
        return matrix_from_numpy(array);
    }
    if (buf.ndim != 1) {
        throw py::value_error("Unsupported array rank for elementwise operation");
    }

    // For elementwise ops, treat 1D NumPy arrays as row vectors (1, n) to match NumPy
    // broadcasting rules when combining with (m, n) matrices.
    if (is_numpy_float16(array)) {
        uint64_t cols = static_cast<uint64_t>(buf.shape[0]);
        auto result = std::make_shared<DenseMatrix<pycauset::float16_t>>(1, cols);

        const uint16_t* src_ptr = static_cast<const uint16_t*>(buf.ptr);
        pycauset::float16_t* dst_ptr = result->data();

        if (buf.strides[0] == static_cast<py::ssize_t>(sizeof(uint16_t))) {
            std::memcpy(static_cast<void*>(dst_ptr), src_ptr, cols * sizeof(uint16_t));
        } else {
            const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
            for (uint64_t j = 0; j < cols; ++j) {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * stride0;
                dst_ptr[j] = pycauset::float16_t(static_cast<uint16_t>(src_ptr[idx]));
            }
        }
        return result;
    }
    if (py::isinstance<py::array_t<double>>(array)) {
        return dense_row_matrix_from_numpy_1d<double>(array.cast<py::array_t<double>>());
    }
    if (py::isinstance<py::array_t<float>>(array)) {
        return dense_row_matrix_from_numpy_1d<float>(array.cast<py::array_t<float>>());
    }
    if (py::isinstance<py::array_t<int32_t>>(array)) {
        return dense_row_matrix_from_numpy_1d<int32_t>(array.cast<py::array_t<int32_t>>());
    }
    if (py::isinstance<py::array_t<int64_t>>(array)) {
        return dense_row_matrix_from_numpy_1d<int64_t>(array.cast<py::array_t<int64_t>>());
    }
    if (py::isinstance<py::array_t<int8_t>>(array)) {
        return dense_row_matrix_from_numpy_1d<int8_t>(array.cast<py::array_t<int8_t>>());
    }
    if (py::isinstance<py::array_t<int16_t>>(array)) {
        return dense_row_matrix_from_numpy_1d<int16_t>(array.cast<py::array_t<int16_t>>());
    }
    if (py::isinstance<py::array_t<uint8_t>>(array)) {
        return dense_row_matrix_from_numpy_1d<uint8_t>(array.cast<py::array_t<uint8_t>>());
    }
    if (py::isinstance<py::array_t<uint16_t>>(array)) {
        return dense_row_matrix_from_numpy_1d<uint16_t>(array.cast<py::array_t<uint16_t>>());
    }
    if (py::isinstance<py::array_t<uint32_t>>(array)) {
        return dense_row_matrix_from_numpy_1d<uint32_t>(array.cast<py::array_t<uint32_t>>());
    }
    if (py::isinstance<py::array_t<uint64_t>>(array)) {
        return dense_row_matrix_from_numpy_1d<uint64_t>(array.cast<py::array_t<uint64_t>>());
    }
    if (py::isinstance<py::array_t<bool>>(array)) {
        auto tmp = array.cast<py::array_t<bool>>();
        auto tmp_buf = tmp.request();
        require_1d(tmp_buf);
        uint64_t cols = static_cast<uint64_t>(tmp_buf.shape[0]);
        auto result = std::make_shared<DenseBitMatrix>(1, cols);
        const bool* src_ptr = static_cast<const bool*>(tmp_buf.ptr);
        const ptrdiff_t stride0 = static_cast<ptrdiff_t>(tmp_buf.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
        for (uint64_t j = 0; j < cols; ++j) {
            const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * stride0;
            result->set(0, j, src_ptr[idx]);
        }
        return result;
    }
    if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
        return dense_row_matrix_from_numpy_1d<std::complex<float>>(
            array.cast<py::array_t<std::complex<float>>>());
    }
    if (py::isinstance<py::array_t<std::complex<double>>>(array)) {
        return dense_row_matrix_from_numpy_1d<std::complex<double>>(
            array.cast<py::array_t<std::complex<double>>>());
    }
    throw py::type_error("Unsupported NumPy dtype for matrix conversion");
}

template <typename Fn>
inline auto translate_invalid_argument(Fn&& fn) {
    try {
        return fn();
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
}

}  // namespace pycauset_bindings::numpy_import
