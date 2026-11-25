#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include <cmath>
#include <limits>
#include "bindings_common.hpp"
#include "element_conversion.hpp"
#include "scalar_args.hpp"
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <iostream>

#include "binding_warnings.hpp"

#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/compute/ComputeContext.hpp"
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

#include "numpy_export_guard.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/ParallelUtils.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

using namespace pycauset;

namespace {

struct DenseMatrix2DBufferState {
    Py_ssize_t shape[2];
    Py_ssize_t strides[2];
};

inline bool u64_to_py_ssize(uint64_t v, Py_ssize_t* out) {
    if (v > static_cast<uint64_t>(PY_SSIZE_T_MAX)) {
        return false;
    }
    *out = static_cast<Py_ssize_t>(v);
    return true;
}

inline bool add_overflow_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (a > (std::numeric_limits<uint64_t>::max)() - b) {
        return true;
    }
    *out = a + b;
    return false;
}

inline bool is_numpy_float16(const py::array& array) {
    try {
        return py::str(array.dtype()).cast<std::string>() == "float16";
    } catch (...) {
        return false;
    }
}

inline bool is_dense_int32_matrix(const MatrixBase& m) {
    return dynamic_cast<const DenseMatrix<int32_t>*>(&m) != nullptr;
}

inline void maybe_warn_integer_matmul_overflow_risk_preflight(const MatrixBase& a, const MatrixBase& b) {
    // ADVISORY ONLY (policy: guesses may inform, never decide). Samples a fixed,
    // deterministic set of entries to estimate max|A|, max|B| and warns early when the
    // result may overflow int32. It never changes the result: the matmul kernel uses an
    // int64 accumulator and raises OverflowError exactly when a value doesn't fit.
    // Value bounds (review/features/VALUE_BOUNDS.md) will make this exact when known.
    auto* ai = dynamic_cast<const DenseMatrix<int32_t>*>(&a);
    auto* bi = dynamic_cast<const DenseMatrix<int32_t>*>(&b);
    if (!ai || !bi) {
        return;
    }

    const uint64_t m = a.rows();
    const uint64_t k = a.cols();
    const uint64_t n = b.cols();
    if (m < 128 || k < 128 || n < 128) {
        return;
    }

    const uint64_t sample_count = std::min<uint64_t>(std::min<uint64_t>(std::min<uint64_t>(m, k), n), 64);
    int64_t max_abs_a = 0;
    int64_t max_abs_b = 0;

    auto update_max_abs = [](int64_t& dst, int32_t v) {
        int64_t av = static_cast<int64_t>(v);
        if (av < 0) av = -av;
        if (av > dst) dst = av;
    };

    // Always include a few fixed positions for determinism.
    update_max_abs(max_abs_a, ai->get(0, 0));
    update_max_abs(max_abs_b, bi->get(0, 0));
    update_max_abs(max_abs_a, ai->get(m - 1, k - 1));
    update_max_abs(max_abs_b, bi->get(k - 1, n - 1));
    update_max_abs(max_abs_a, ai->get(0, k - 1));
    update_max_abs(max_abs_b, bi->get(k - 1, 0));
    update_max_abs(max_abs_a, ai->get(m - 1, 0));
    update_max_abs(max_abs_b, bi->get(0, n - 1));

    // Stride-based sample (no RNG; stable across runs).
    // Uses Knuth multiplicative hashing to spread indices.
    for (uint64_t s = 0; s < sample_count; ++s) {
        uint64_t i = (s * 2654435761u) % m;
        uint64_t j = (s * 805459861u + 1u) % k;
        uint64_t l = (s * 11400714819323198485ull + 2u) % n;
        update_max_abs(max_abs_a, ai->get(i, j));
        update_max_abs(max_abs_b, bi->get(j, l));
    }

    if (max_abs_a == 0 || max_abs_b == 0) {
        return;
    }

    const double scalar_factor = std::abs(a.get_scalar()) * std::abs(b.get_scalar());
    const double bound = static_cast<double>(k) *
                         static_cast<double>(max_abs_a) *
                         static_cast<double>(max_abs_b) *
                         scalar_factor;

    const double out_max = static_cast<double>((std::numeric_limits<int32_t>::max)());
    if (!(bound > out_max)) {
        return;
    }

    std::string message;
    message.reserve(512);
    message += "pycauset matmul: int32 @ int32 may overflow the int32 output (advisory estimate from sampled entries). ";
    message += "m=" + std::to_string(m);
    message += ", k=" + std::to_string(k);
    message += ", n=" + std::to_string(n);
    message += ", sampled max|A|=" + std::to_string(max_abs_a);
    message += ", sampled max|B|=" + std::to_string(max_abs_b);
    message += ", estimated bound ~" + std::to_string(bound);
    message += ". The result is computed exactly; PyCauset raises OverflowError if a value does not fit.";
    bindings_warn::warn_with_category(message,
        "PyCausetOverflowRiskWarning",
        /*stacklevel=*/3);
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

inline void require_square_2d(const py::buffer_info& buf) {
    if (buf.ndim != 2) {
        throw py::value_error("Number of dimensions must be 2");
    }
    if (buf.shape[0] != buf.shape[1]) {
        throw py::value_error("Matrix must be square");
    }
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

// Helper for Parallel Export
template <typename T>
py::array_t<T> parallel_export_to_numpy(const DenseMatrix<T>& mat) {
    // 1. Check Safety (Guard)
    pycauset::bindings::ensure_numpy_export_allowed(static_cast<const MatrixBase&>(mat));

    uint64_t rows = mat.rows();
    uint64_t cols = mat.cols();
    const bool needs_slow_path = mat.is_transposed() || mat.has_view_offset() ||
        mat.is_conjugated() || mat.get_scalar() != std::complex<double>(1.0, 0.0);
    if (needs_slow_path) {
        py::array_t<T> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
        auto r = out.template mutable_unchecked<2>();
        for (uint64_t i = 0; i < rows; ++i) {
            for (uint64_t j = 0; j < cols; ++j) {
                if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                    const std::complex<double> v = mat.get_element_as_complex(i, j);
                    r(i, j) = static_cast<T>(v);
                } else {
                    const double v = mat.get_element_as_double(i, j);
                    r(i, j) = static_cast<T>(v);
                }
            }
        }
        return out;
    }
    size_t total_bytes = rows * cols * sizeof(T);

    // 2. Allocate uninitialized NumPy array
    // We use C-contiguous order by default for new arrays.
    // Use numpy.empty explicitly to avoid any risk of zero-initialization overhead
    auto np = py::module::import("numpy");
    py::object result_obj = np.attr("empty")(py::make_tuple(rows, cols), py::dtype::of<T>());
    py::array_t<T> result = result_obj.cast<py::array_t<T>>();

    // 3. Parallel Copy
    // If small, use single threaded to avoid overhead.
    // If huge, use ParallelFor with GIL release.
    constexpr size_t PARALLEL_THRESHOLD_BYTES = 1024 * 1024; // 1MB

    py::buffer_info buf = result.request();
    T* dst_ptr = static_cast<T*>(buf.ptr);
    const T* src_ptr = mat.data();

    // Check strict C-contiguous for safety (newly allocated numpy array usually is)
    // bool is_c_contiguous = (buf.strides[1] == sizeof(T) && buf.strides[0] == static_cast<py::ssize_t>(cols * sizeof(T)));
    
    if (total_bytes >= PARALLEL_THRESHOLD_BYTES) {
        // Release GIL for parallel copy
        {
            py::gil_scoped_release release;
            // Use strides from buffer info to be safe, though we expect C-contiguous.
            const py::ssize_t stride0 = buf.strides[0];
            char* dst_base = static_cast<char*>(buf.ptr);

            // DEBUG PRINT
            // printf("Parallel Export: rows=%llu, cols=%llu, stride0=%lld, threads=%zu\n",
            //        (unsigned long long)rows, (unsigned long long)cols, (long long)stride0, pycauset::ThreadPool::get_num_threads());
            
            pycauset::ParallelFor(0, rows, [&](size_t i) {
                const T* src_row = src_ptr + i * cols;
                // Calculate destination row using stride
                T* dst_row = reinterpret_cast<T*>(dst_base + i * stride0);
                std::memcpy(dst_row, src_row, cols * sizeof(T));
            });
        }
    } else {
        // Fallback: Safe single-threaded consistency
        std::memcpy(dst_ptr, src_ptr, total_bytes);
    }

    return result;
}

template <typename T>
std::shared_ptr<DenseMatrix<T>> dense_matrix_from_numpy_2d(const py::array_t<T>& array) {
    auto buf = array.request();
    require_2d(buf);

    uint64_t rows = static_cast<uint64_t>(buf.shape[0]);
    uint64_t cols = static_cast<uint64_t>(buf.shape[1]);

    // R1_NUMPY: Check MemoryGovernor before allocating.
    // If the matrix is too large for RAM, we should fail fast rather than OOM/Swap-death.
    // Note: The input is already in RAM (NumPy), so we are effectively doubling memory usage here.
    size_t required_bytes = rows * cols * sizeof(T);
    if (!pycauset::core::MemoryGovernor::instance().should_use_direct_path(required_bytes)) {
        throw std::runtime_error("Import rejected by MemoryGovernor: Matrix too large for available RAM.");
    }

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
    // Normalize to exactly two entries (row, col). Ellipsis expands to full slices.
    py::object first;
    py::object second;

    if (py::isinstance<py::ellipsis>(key)) {
        first = py::slice();
        second = py::slice();
    } else if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
        py::sequence seq = key.cast<py::sequence>();
        if (seq.size() == 1) {
            if (py::isinstance<py::ellipsis>(seq[0])) {
                first = py::slice();
                second = py::slice();
            } else {
                throw py::type_error("Matrix indices must be (row, col)");
            }
        } else if (seq.size() == 2) {
            first = seq[0];
            second = seq[1];
            if (py::isinstance<py::ellipsis>(first)) {
                first = py::slice();
            }
            if (py::isinstance<py::ellipsis>(second)) {
                second = py::slice();
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

inline std::pair<uint64_t, uint64_t> parse_matrix_index(const py::handle& key, uint64_t rows, uint64_t cols) {
    auto parsed = parse_matrix_subscript(key, rows, cols);
    if (!parsed.rows.is_scalar || !parsed.cols.is_scalar) {
        throw py::type_error("Matrix indices must be integers (no slices) for this operation");
    }
    return {parsed.rows.index, parsed.cols.index};
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
                                                         mat.is_conjugated(),
                                                         mat.is_temporary());
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
        return result;
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
        return result;
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
        return result;
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
    return result;
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

inline bool is_complex_dtype(DataType dt) {
    return dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64;
}

inline bool is_real_float_dtype(DataType dt) {
    return dt == DataType::FLOAT16 || dt == DataType::FLOAT32 || dt == DataType::FLOAT64;
}

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

// Before writing into a view that carries a scale factor or conjugation, fold the view
// into private storage so a write reads back exactly what was written (F-045). Dense only;
// other structures with a factor raise instead of storing a wrong value (lazy L1 replaces
// this with ViewSpec-aware writes).
inline void fold_value_view_for_write(MatrixBase& mat) {
    const std::complex<double> s = mat.get_scalar();
    const bool conj = mat.is_conjugated();
    if (s == std::complex<double>(1.0, 0.0) && !conj) return;
    if (mat.get_structure() != pycauset::Structure::DENSE) {
        // Structured (real-valued) storage: read every stored element through the view, then
        // rewrite it with the factor cleared. Positions follow the logical structure.
        using S = pycauset::Structure;
        const uint64_t n = mat.rows();
        const auto st = mat.get_structure();
        if (st == S::IDENTITY) throw py::type_error(describe(mat) + " is read-only");
        bool lower = false, diag_too = false;
        if (st == S::TRIANGULAR) {
            lower = mat.is_transposed();
            diag_too = dynamic_cast<const TriangularMatrixBase&>(mat).has_diagonal();
        }
        auto stored = [&](uint64_t i, uint64_t j) {
            switch (st) {
                case S::DIAGONAL: return i == j;
                case S::SYMMETRIC: return j >= i;
                case S::ANTISYMMETRIC: return j > i;
                case S::TRIANGULAR: return lower ? (i > j || (diag_too && i == j)) : (j > i || (diag_too && i == j));
                default: return false;
            }
        };
        pycauset::visit_matrix(mat, [&](auto& typed) {
            using E = element_of_t<std::decay_t<decltype(typed)>>;
            if constexpr (requires { typed.set(uint64_t{0}, uint64_t{0}, E{}); }) {
                // Read every stored element through the view (exactly for integers), clear the
                // factor, write back.
                std::vector<std::tuple<uint64_t, uint64_t, E>> values;
                const int64_t k = static_cast<int64_t>(s.real());
                for (uint64_t i = 0; i < n; ++i)
                    for (uint64_t j = 0; j < n; ++j)
                        if (stored(i, j)) {
                            if constexpr (std::is_integral_v<E> && !std::is_same_v<E, bool>) {
                                values.emplace_back(i, j, pycauset::scalar::wrap_scale(typed.get(i, j), k));
                            } else {
                                values.emplace_back(i, j, static_cast<E>(mat.get_element_as_double(i, j)));
                            }
                        }
                mat.set_scalar(1.0);
                mat.set_conjugated(false);
                for (const auto& [i, j, v] : values) typed.set(i, j, v);
            }
        });
        return;
    }
    pycauset::visit_matrix(mat, [&](auto& typed) {
        using E = element_of_t<std::decay_t<decltype(typed)>>;
        if constexpr (std::is_same_v<E, bool>) {
            throw std::logic_error("bit matrix carries a scale factor");
        } else if constexpr (std::is_integral_v<E>) {
            const int64_t k = static_cast<int64_t>(s.real());
            const uint64_t r = typed.rows(), c = typed.cols();
            for (uint64_t i = 0; i < r; ++i)
                for (uint64_t j = 0; j < c; ++j) typed.set(i, j, pycauset::scalar::wrap_scale(typed.get(i, j), k));
        } else {
            const uint64_t r = typed.rows(), c = typed.cols();
            for (uint64_t i = 0; i < r; ++i) {
                for (uint64_t j = 0; j < c; ++j) {
                    if constexpr (std::is_same_v<E, std::complex<double>> || std::is_same_v<E, std::complex<float>>) {
                        std::complex<double> v = pycauset::scalar::to_complex_double(typed.get(i, j)) * s;
                        if (conj) v = std::conj(v);
                        typed.set(i, j, static_cast<E>(v));
                    } else if constexpr (std::is_same_v<std::decay_t<decltype(typed)>, pycauset::ComplexFloat16Matrix>) {
                        std::complex<double> v = typed.get(i, j) * s;
                        if (conj) v = std::conj(v);
                        typed.set(i, j, v);
                    } else {
                        const double v = (pycauset::scalar::to_complex_double(typed.get(i, j)) * s).real();
                        typed.set(i, j, static_cast<E>(v));
                    }
                }
            }
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

inline std::shared_ptr<MatrixBase> matrix_inverse(const MatrixBase& mat) {
    return pycauset::visit_matrix(mat, [&](const auto& typed) -> std::shared_ptr<MatrixBase> {
        using M = std::decay_t<decltype(typed)>;
        // DenseMatrix<bool>::inverse is declared but has no definition.
        if constexpr (!std::is_same_v<M, DenseBitMatrix> && requires { typed.inverse(std::string()); }) {
            return std::shared_ptr<MatrixBase>(typed.inverse("").release());
        } else {
            throw py::type_error("inverse is not supported for " + describe(mat));
        }
    });
}

inline std::shared_ptr<MatrixBase> bit_binary(const MatrixBase& a, const MatrixBase& b, char op) {
    const auto* x = dynamic_cast<const DenseBitMatrix*>(&a);
    const auto* y = dynamic_cast<const DenseBitMatrix*>(&b);
    if (x == nullptr || y == nullptr) {
        throw py::type_error(std::string("bitwise ") + op + " is supported for dense matrices of dtype 'bool'");
    }
    if (x->rows() != y->rows() || x->cols() != y->cols()) {
        throw py::value_error("matrix dimensions must match for bitwise operations");
    }
    auto out = std::make_shared<DenseBitMatrix>(x->rows(), x->cols(), "");
    // Word-level fast path for plain storage (64 elements per operation); lost in the
    // structure-class rewrite and restored (F-054).
    if (!x->is_transposed() && !y->is_transposed() && !x->has_view_offset() && !y->has_view_offset() &&
        x->rows() == x->base_rows() && x->cols() == x->base_cols() &&
        y->rows() == y->base_rows() && y->cols() == y->base_cols() && x->shared_mapper() && y->shared_mapper()) {
        const uint64_t* pa = x->data();
        const uint64_t* pb = y->data();
        uint64_t* pcw = out->data();
        const uint64_t words = x->stride_bytes() * x->rows() / 8;
        pycauset::ParallelFor(0, words, [&](size_t w) {
            pcw[w] = op == '&' ? (pa[w] & pb[w]) : op == '|' ? (pa[w] | pb[w]) : (pa[w] ^ pb[w]);
        });
        return out;
    }
    for (uint64_t i = 0; i < x->rows(); ++i) {
        for (uint64_t j = 0; j < x->cols(); ++j) {
            const bool p = x->get(i, j), q = y->get(i, j);
            out->set(i, j, op == '&' ? (p && q) : op == '|' ? (p || q) : (p != q));
        }
    }
    return out;
}

// Construct a structure of a dtype given by name, through the one factory.
template <typename StructureBase>
std::shared_ptr<StructureBase> make_structure(uint64_t rows, uint64_t cols, pycauset::Structure s, const std::string& dtype) {
    auto obj = pycauset::ObjectFactory::create_matrix(rows, cols, pycauset::parse_dtype_name(dtype), s, "");
    auto* typed = dynamic_cast<StructureBase*>(obj.get());
    if (typed == nullptr) throw std::logic_error("factory returned the wrong structure");
    obj.release();
    return std::shared_ptr<StructureBase>(typed);
}

inline py::object symmetric_from_triangular(const MatrixBase& source, bool anti) {
    if (source.get_structure() != pycauset::Structure::TRIANGULAR) {
        throw py::type_error("from_triangular expects a TriangularMatrix");
    }
    return pycauset::visit_matrix(source, [&](const auto& tri) -> py::object {
        using M = std::decay_t<decltype(tri)>;
        using E = element_of_t<M>;
        if constexpr (std::is_same_v<M, TriangularMatrix<E>> && !std::is_same_v<E, bool>) {
            if (anti) return py::cast(std::shared_ptr<MatrixBase>(AntiSymmetricMatrix<E>::from_triangular(tri, "").release()));
            return py::cast(std::shared_ptr<MatrixBase>(SymmetricMatrix<E>::from_triangular(tri, "").release()));
        } else {
            throw py::type_error("from_triangular is not supported for " + describe(source));
        }
    });
}

// Re-open stored data: the constructors take the *storage* shape and the view flags.
inline py::object open_storage(pycauset::Structure s, DataType dt, uint64_t rows, uint64_t cols,
                               const std::string& path, size_t offset, uint64_t seed,
                               std::complex<double> scalar, bool is_transposed) {
    using S = pycauset::Structure;
    return pycauset::visit_dtype(dt, [&](auto tag) -> py::object {
        using T = typename decltype(tag)::type;
        if (s == S::DENSE) {
            using M = typename pycauset::DenseMatrixOf<T>::type;
            return py::cast(std::shared_ptr<MatrixBase>(std::make_shared<M>(rows, cols, path, offset, seed, scalar, is_transposed)));
        }
        if (s == S::VECTOR) {
            using V = typename pycauset::DenseVectorOf<T>::type;
            return py::cast(std::shared_ptr<VectorBase>(std::make_shared<V>(rows, path, offset, seed, scalar, is_transposed)));
        }
        if constexpr (pycauset::is_complex_element_v<T>) {
            pycauset::throw_unsupported(s, dt);
        } else {
            switch (s) {
                case S::TRIANGULAR:
                    return py::cast(std::shared_ptr<MatrixBase>(std::make_shared<TriangularMatrix<T>>(rows, path, offset, seed, scalar, is_transposed)));
                case S::SYMMETRIC:
                case S::ANTISYMMETRIC:
                    return py::cast(std::shared_ptr<MatrixBase>(std::make_shared<SymmetricMatrix<T>>(rows, path, offset, seed, scalar, is_transposed, s == S::ANTISYMMETRIC)));
                case S::DIAGONAL:
                    if constexpr (std::is_same_v<T, pycauset::float16_t> || std::is_same_v<T, float>) pycauset::throw_unsupported(s, dt);
                    else return py::cast(std::shared_ptr<MatrixBase>(std::make_shared<DiagonalMatrix<T>>(rows, path, offset, seed, scalar, is_transposed)));
                case S::IDENTITY:
                    if constexpr (std::is_same_v<T, pycauset::float16_t> || std::is_same_v<T, float>) pycauset::throw_unsupported(s, dt);
                    else return py::cast(std::shared_ptr<MatrixBase>(std::make_shared<IdentityMatrix<T>>(rows, cols, path, offset, seed, scalar, is_transposed)));
                case S::UNIT_VECTOR:
                    return py::cast(std::shared_ptr<VectorBase>(std::make_shared<UnitVector>(rows, seed, path, offset, seed, scalar, is_transposed)));
                default: pycauset::throw_unsupported(s, dt);
            }
        }
    });
}

template <typename Fn>
inline auto translate_invalid_argument(Fn&& fn) {
    try {
        return fn();
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
}

inline bool is_float_dtype(const DataType dt) {
    return dt == DataType::FLOAT16 || dt == DataType::FLOAT32 || dt == DataType::FLOAT64;
}



// ---------------------------------------------------------------------------
// Native NumPy export for dense matrices of any dtype (used by _to_numpy_fast).
// Contiguous, unscaled, non-transposed storage is memcpy'd; otherwise elements are
// read exactly through the typed accessor (float/complex apply the lazy scalar and
// conjugation through the virtual accessors). Returns None if `mat` isn't dense.
// ---------------------------------------------------------------------------
template <typename T, typename NpT = T>
py::object export_dense_typed(const MatrixBase& mat) {
    auto* m = dynamic_cast<const DenseMatrix<T>*>(&mat);
    if (!m) return py::none();
    const uint64_t rows = mat.rows(), cols = mat.cols();
    py::array_t<NpT> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    const bool unit_view = mat.get_scalar() == std::complex<double>(1.0, 0.0) && !mat.is_conjugated();
    if (unit_view && !mat.is_transposed() && !mat.has_view_offset() && mat.shared_mapper()) {
        if (const void* src = mat.shared_mapper()->get_data()) {
            std::memcpy(static_cast<void*>(out.request().ptr), src, rows * cols * sizeof(T));
            return out;
        }
    }
    auto r = out.template mutable_unchecked<2>();
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                // Integer views (F-042): factor applied with NumPy's wrapping.
                r(i, j) = unit_view ? m->get(i, j)
                                    : pycauset::scalar::wrap_scale(m->get(i, j), static_cast<int64_t>(mat.get_scalar().real()));
            } else if constexpr (std::is_integral_v<T>) {
                r(i, j) = m->get(i, j);
            } else if constexpr (std::is_same_v<T, pycauset::float16_t>) {
                const double v = m->get_element_as_double(i, j);
                r(i, j) = pycauset::float16_t(v).bits;
            } else if constexpr (std::is_floating_point_v<T>) {
                r(i, j) = static_cast<T>(m->get_element_as_double(i, j));
            } else {
                const std::complex<double> v = m->get_element_as_complex(i, j);
                r(i, j) = T(static_cast<typename T::value_type>(v.real()), static_cast<typename T::value_type>(v.imag()));
            }
        }
    }
    return out;
}

inline py::object export_dense_any(const MatrixBase& mat) {
    switch (mat.get_data_type()) {
        case DataType::INT8: return export_dense_typed<int8_t>(mat);
        case DataType::INT16: return export_dense_typed<int16_t>(mat);
        case DataType::INT32: return export_dense_typed<int32_t>(mat);
        case DataType::INT64: return export_dense_typed<int64_t>(mat);
        case DataType::UINT8: return export_dense_typed<uint8_t>(mat);
        case DataType::UINT16: return export_dense_typed<uint16_t>(mat);
        case DataType::UINT32: return export_dense_typed<uint32_t>(mat);
        case DataType::UINT64: return export_dense_typed<uint64_t>(mat);
        case DataType::FLOAT32: return export_dense_typed<float>(mat);
        case DataType::FLOAT64: return export_dense_typed<double>(mat);
        case DataType::FLOAT16: {
            py::object raw = export_dense_typed<pycauset::float16_t, uint16_t>(mat);
            return raw.is_none() ? raw : raw.attr("view")(py::module_::import("numpy").attr("float16"));
        }
        case DataType::COMPLEX_FLOAT32: return export_dense_typed<std::complex<float>>(mat);
        case DataType::COMPLEX_FLOAT64: return export_dense_typed<std::complex<double>>(mat);
        default: return py::none();
    }
}

// Any structure / dtype (triangular, symmetric, diagonal, identity, closures, bits,
// complex16): read every logical element in C++ (exact for integers, view-aware) into a new
// NumPy array. Replaces a Python per-element loop (~100x slower).
inline py::object export_generic(const MatrixBase& mat) {
    const uint64_t rows = mat.rows(), cols = mat.cols();
    const DataType dt = mat.get_data_type();
    const char* np_name = nullptr;
    switch (dt) {
        case DataType::BIT: np_name = "bool"; break;
        case DataType::INT8: np_name = "int8"; break;
        case DataType::INT16: np_name = "int16"; break;
        case DataType::INT32: np_name = "int32"; break;
        case DataType::INT64: np_name = "int64"; break;
        case DataType::UINT8: np_name = "uint8"; break;
        case DataType::UINT16: np_name = "uint16"; break;
        case DataType::UINT32: np_name = "uint32"; break;
        case DataType::UINT64: np_name = "uint64"; break;
        case DataType::FLOAT16: np_name = "float16"; break;
        case DataType::FLOAT32: np_name = "float32"; break;
        case DataType::FLOAT64: np_name = "float64"; break;
        case DataType::COMPLEX_FLOAT16:
        case DataType::COMPLEX_FLOAT32: np_name = "complex64"; break;
        case DataType::COMPLEX_FLOAT64: np_name = "complex128"; break;
        default: return py::none();
    }
    py::array out(py::dtype(np_name), std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    void* dst = out.mutable_data();
    if (pycauset::is_integer_or_bit(dt)) {
        const bool scaled = mat.get_scalar() != std::complex<double>(1.0, 0.0);
        if (scaled && !pycauset::scalar::is_exact_int_factor(mat.get_scalar())) return py::none();
        const int64_t k = static_cast<int64_t>(mat.get_scalar().real());
        py::gil_scoped_release release;
        pycauset::visit_matrix(mat, [&](const auto& typed) {
            using E = element_of_t<std::decay_t<decltype(typed)>>;
            if constexpr (std::is_integral_v<E>) {
                auto* p = static_cast<E*>(dst);
                pycauset::ParallelFor(0, rows, [&](size_t i) {
                    for (uint64_t j = 0; j < cols; ++j) {
                        const E v = typed.get(i, j);
                        p[i * cols + j] = scaled ? pycauset::scalar::wrap_scale(v, k) : v;
                    }
                });
            }
        });
        return out;
    }
    py::gil_scoped_release release;
    if (dt == DataType::FLOAT16) {
        auto* p = static_cast<uint16_t*>(dst);
        pycauset::ParallelFor(0, rows, [&](size_t i) {
            for (uint64_t j = 0; j < cols; ++j) p[i * cols + j] = pycauset::float16_t(mat.get_element_as_double(i, j)).bits;
        });
    } else if (dt == DataType::FLOAT32) {
        auto* p = static_cast<float*>(dst);
        pycauset::ParallelFor(0, rows, [&](size_t i) {
            for (uint64_t j = 0; j < cols; ++j) p[i * cols + j] = static_cast<float>(mat.get_element_as_double(i, j));
        });
    } else if (dt == DataType::FLOAT64) {
        auto* p = static_cast<double*>(dst);
        pycauset::ParallelFor(0, rows, [&](size_t i) {
            for (uint64_t j = 0; j < cols; ++j) p[i * cols + j] = mat.get_element_as_double(i, j);
        });
    } else if (dt == DataType::COMPLEX_FLOAT64) {
        auto* p = static_cast<std::complex<double>*>(dst);
        pycauset::ParallelFor(0, rows, [&](size_t i) {
            for (uint64_t j = 0; j < cols; ++j) p[i * cols + j] = mat.get_element_as_complex(i, j);
        });
    } else {
        auto* p = static_cast<std::complex<float>*>(dst);
        pycauset::ParallelFor(0, rows, [&](size_t i) {
            for (uint64_t j = 0; j < cols; ++j) p[i * cols + j] = static_cast<std::complex<float>>(mat.get_element_as_complex(i, j));
        });
    }
    return out;
}



// Matrix arithmetic, the single operator path (PLAN 4.3): `a op other` (`other op a` when
// reflected) for another matrix, any Python/NumPy scalar, or a NumPy array. Scalars go to
// scalar_op (NumPy result dtypes, exact integers, O(1) float views); matrices to the
// element-wise ops. Anything else returns NotImplemented (e.g. BlockMatrix handles it).
inline py::object matrix_arith(const MatrixBase& a, const py::object& other, Arith op, bool reflected) {
    std::shared_ptr<MatrixBase> b;
    if (py::isinstance<MatrixBase>(other)) {
        b = other.cast<std::shared_ptr<MatrixBase>>();
    } else if (pycauset::ScalarArg s; to_scalar_arg(other, s)) {
        return py::cast(std::shared_ptr<MatrixBase>(pycauset::scalar_op(a, scalar_kind(op, reflected), s).release()));
    } else if (py::isinstance<py::array>(other)) {
        b = elementwise_operand_matrix_from_numpy(other.cast<py::array>());
    } else {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    const MatrixBase& x = reflected ? *b : a;
    const MatrixBase& y = reflected ? a : *b;
    std::unique_ptr<MatrixBase> out;
    switch (op) {
        case Arith::Add:
            maybe_warn_float_underpromotion("add", promotion::BinaryOp::Add, x, y);
            maybe_warn_bit_promotes_to_int32("add", promotion::BinaryOp::Add, x.get_data_type(), y.get_data_type());
            out = pycauset::add(x, y, "");
            break;
        case Arith::Sub:
            maybe_warn_float_underpromotion("subtract", promotion::BinaryOp::Subtract, x, y);
            maybe_warn_bit_promotes_to_int32("subtract", promotion::BinaryOp::Subtract, x.get_data_type(), y.get_data_type());
            out = pycauset::subtract(x, y, "");
            break;
        case Arith::Mul:
            maybe_warn_float_underpromotion("elementwise_multiply", promotion::BinaryOp::ElementwiseMultiply, x, y);
            out = pycauset::elementwise_multiply(x, y, "");
            break;
        case Arith::Div:
            maybe_warn_float_underpromotion("elementwise_divide", promotion::BinaryOp::Divide, x, y);
            out = pycauset::elementwise_divide(x, y, "");
            break;
    }
    return py::cast(std::shared_ptr<MatrixBase>(out.release()));
}

} // namespace

void bind_matrix_classes(py::module_& m) {
    auto matrix_base_class = py::class_<MatrixBase, std::shared_ptr<MatrixBase>>(m, "MatrixBase", py::dynamic_attr());
    matrix_base_class
        .def_property_readonly("__array_priority__", [](const MatrixBase&) { 
            return 1000.0; 
        })
        .def_property_readonly("backing_file", &MatrixBase::get_backing_file)
        .def("get_backing_file", &MatrixBase::get_backing_file)
        .def_property_readonly("is_temporary", &MatrixBase::is_temporary)
        .def("set_temporary", &MatrixBase::set_temporary)
        .def("close", &MatrixBase::close)
        .def("copy_storage", &MatrixBase::copy_storage, py::arg("result_file_hint") = "")
        .def(
            "get_accelerator",
            &MatrixBase::get_accelerator,
            py::return_value_policy::reference_internal)
        .def("hint", &MatrixBase::hint, py::arg("hint"))
        .def_property("seed", &MatrixBase::get_seed, &MatrixBase::set_seed)
        .def_property("scalar", &MatrixBase::get_scalar, &MatrixBase::set_scalar)
        .def("get_scalar", &MatrixBase::get_scalar)
        .def("set_scalar", &MatrixBase::set_scalar)
        .def("is_transposed", &MatrixBase::is_transposed)
        .def("set_transposed", &MatrixBase::set_transposed)
        .def("is_conjugated", &MatrixBase::is_conjugated)
        .def("set_conjugated", &MatrixBase::set_conjugated)
        .def("_get_properties_flags", &MatrixBase::get_properties_flags)
        .def("_set_properties_flags", &MatrixBase::set_properties_flags)
        .def("_set_property_flag", &MatrixBase::set_property_flag, py::arg("flag"), py::arg("value"))
        .def_property_readonly(
            "T",
            [](const MatrixBase& mat) {
                auto out = mat.transpose("");
                return std::shared_ptr<MatrixBase>(out.release());
            })
        .def_property_readonly(
            "H",
            [](const MatrixBase& mat) {
                auto out = mat.transpose("");
                out->set_conjugated(!mat.is_conjugated());
                return std::shared_ptr<MatrixBase>(out.release());
            })
        .def(
            "transpose",
            [](const MatrixBase& mat) {
                auto out = mat.transpose("");
                return std::shared_ptr<MatrixBase>(out.release());
            })
        .def(
            "conjugate",
            [](const MatrixBase& mat) {
                auto out = mat.clone();
                auto* m = dynamic_cast<MatrixBase*>(out.get());
                if (!m) {
                    throw std::runtime_error("Internal error: clone() did not return a MatrixBase");
                }
                m->set_conjugated(!mat.is_conjugated());
                return std::shared_ptr<MatrixBase>(static_cast<MatrixBase*>(out.release()));
            })
        .def(
            "conj",
            [](const MatrixBase& mat) {
                auto out = mat.clone();
                auto* m = dynamic_cast<MatrixBase*>(out.get());
                if (!m) {
                    throw std::runtime_error("Internal error: clone() did not return a MatrixBase");
                }
                m->set_conjugated(!mat.is_conjugated());
                return std::shared_ptr<MatrixBase>(static_cast<MatrixBase*>(out.release()));
            })
        .def_property_readonly("shape", [](const MatrixBase& mat) {
            return py::make_tuple(mat.rows(), mat.cols());
        })
        .def("size", &MatrixBase::size)
        .def("rows", &MatrixBase::rows)
        .def("cols", &MatrixBase::cols)
        .def(
            "multiply",
            [](const MatrixBase& a, const std::shared_ptr<MatrixBase>& b) {
                return translate_invalid_argument([&]() {
                    maybe_warn_float_underpromotion("matmul", promotion::BinaryOp::Matmul, a, *b);
                    maybe_warn_bit_promotes_to_int32(
                        "matmul",
                        promotion::BinaryOp::Matmul,
                        a.get_data_type(),
                        b->get_data_type());
                    maybe_warn_integer_matmul_overflow_risk_preflight(a, *b);
                    auto out = pycauset::dispatch_matmul(a, *b, "");
                    return std::shared_ptr<MatrixBase>(out.release());
                });
            },
            py::arg("other"))
        .def("get_element_as_double", &MatrixBase::get_element_as_double)
        .def("get_element_as_complex", &MatrixBase::get_element_as_complex)
        .def("_to_numpy_fast", [](const MatrixBase& mat, bool allow_huge) -> py::object {
            if (!allow_huge) {
                pycauset::bindings::ensure_numpy_export_allowed(mat);
            }
            // One generic exporter for every dense dtype (previously only float64,
            // int32 and complex128 had a native path; the rest fell into a slow Python
            // loop that read 64-bit integers through double and lost precision).
            py::object out = export_dense_any(mat);
            if (!out.is_none()) return out;
            return export_generic(mat);
        }, py::arg("allow_huge") = false)
        .def(
            "__array__",
            [](const MatrixBase& mat, py::object /*dtype*/, py::object /*copy*/) -> py::array {
                pycauset::bindings::ensure_numpy_export_allowed(mat);
                uint64_t rows = mat.rows();
                uint64_t cols = mat.cols();

                const DataType dt = mat.get_data_type();
                if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
                    if (dt == DataType::COMPLEX_FLOAT64) {
                        py::array out(
                            py::dtype("complex128"),
                            py::array::ShapeContainer{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                        auto buf = out.request();
                        auto* dst_ptr = static_cast<std::complex<double>*>(buf.ptr);
                        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(std::complex<double>)));
                        const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(std::complex<double>)));
                        for (uint64_t i = 0; i < rows; ++i) {
                            for (uint64_t j = 0; j < cols; ++j) {
                                dst_ptr[i * stride0 + j * stride1] = mat.get_element_as_complex(i, j);
                            }
                        }
                        return out;
                    }

                    // complex64 for both complex_float16 and complex_float32.
                    py::array out(
                        py::dtype("complex64"),
                        py::array::ShapeContainer{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto buf = out.request();
                    auto* dst_ptr = static_cast<std::complex<float>*>(buf.ptr);
                    const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(std::complex<float>)));
                    const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(std::complex<float>)));
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            const std::complex<double> v = mat.get_element_as_complex(i, j);
                            dst_ptr[i * stride0 + j * stride1] = std::complex<float>(static_cast<float>(v.real()), static_cast<float>(v.imag()));
                        }
                    }
                    return out;
                }

                if (auto* mf16 = dynamic_cast<const DenseMatrix<pycauset::float16_t>*>(&mat)) {
                    py::array out(
                        py::dtype("float16"),
                        py::array::ShapeContainer{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto buf = out.request();
                    auto* dst_ptr = static_cast<uint16_t*>(buf.ptr);
                    const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                    const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            dst_ptr[i * stride0 + j * stride1] = mf16->get(i, j).bits;
                        }
                    }
                    return out;
                }
                if (auto* mf32 = dynamic_cast<const DenseMatrix<float>*>(&mat)) {
                    py::array_t<float> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            (void)mf32;
                            r(i, j) = static_cast<float>(mat.get_element_as_double(i, j));
                        }
                    }
                    return out;
                }
                if (auto* mi32 = dynamic_cast<const DenseMatrix<int32_t>*>(&mat)) {
                    py::array_t<int32_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mi32->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mi64 = dynamic_cast<const DenseMatrix<int64_t>*>(&mat)) {
                    py::array_t<int64_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mi64->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mi8 = dynamic_cast<const DenseMatrix<int8_t>*>(&mat)) {
                    py::array_t<int8_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mi8->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mi16 = dynamic_cast<const DenseMatrix<int16_t>*>(&mat)) {
                    py::array_t<int16_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mi16->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mu8 = dynamic_cast<const DenseMatrix<uint8_t>*>(&mat)) {
                    py::array_t<uint8_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mu8->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mu16 = dynamic_cast<const DenseMatrix<uint16_t>*>(&mat)) {
                    py::array_t<uint16_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mu16->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mu32 = dynamic_cast<const DenseMatrix<uint32_t>*>(&mat)) {
                    py::array_t<uint32_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mu32->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mu64 = dynamic_cast<const DenseMatrix<uint64_t>*>(&mat)) {
                    py::array_t<uint64_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mu64->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mb = dynamic_cast<const DenseBitMatrix*>(&mat)) {
                    py::array_t<bool> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mb->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mtb = dynamic_cast<const TriangularBitMatrix*>(&mat)) {
                    py::array_t<bool> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mtb->get(i, j);
                        }
                    }
                    return out;
                }

                // Default: float64
                py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                auto r = out.template mutable_unchecked<2>();
                for (uint64_t i = 0; i < rows; ++i) {
                    for (uint64_t j = 0; j < cols; ++j) {
                        r(i, j) = mat.get_element_as_double(i, j);
                    }
                }
                return out;
            },
            py::arg("dtype") = py::none(),
            py::arg("copy") = py::none())

        .def("trace", [](py::object self) {
            auto& mat = self.cast<MatrixBase&>();
            double v = pycauset::trace(mat);
            return v;
        })
        .def("determinant", [](py::object self) {
            auto& mat = self.cast<MatrixBase&>();
            double v = pycauset::determinant(mat);
            return v;
        })
        .def(
            "__matmul__",
            [](const MatrixBase& a, const std::shared_ptr<MatrixBase>& b) {
                return translate_invalid_argument([&]() {
                    maybe_warn_float_underpromotion("matmul", promotion::BinaryOp::Matmul, a, *b);
                    maybe_warn_bit_promotes_to_int32(
                        "matmul",
                        promotion::BinaryOp::Matmul,
                        a.get_data_type(),
                        b->get_data_type());
                    maybe_warn_integer_matmul_overflow_risk_preflight(a, *b);
                    auto out = pycauset::dispatch_matmul(a, *b, "");
                    return std::shared_ptr<MatrixBase>(out.release());
                });
            },
            py::is_operator())
        .def(
            "__matmul__",
            [](const MatrixBase& a, const std::shared_ptr<VectorBase>& v) {
                return translate_invalid_argument([&]() {
                    maybe_warn_bit_promotes_to_int32(
                        "matmul",
                        promotion::BinaryOp::MatrixVectorMultiply,
                        a.get_data_type(),
                        v->get_data_type());
                    auto out = pycauset::matrix_vector_multiply(a, *v, "");
                    return std::shared_ptr<VectorBase>(out.release());
                });
            },
            py::is_operator())
        .def(
            "__matmul__",
            [](const MatrixBase& a, const py::array& b) -> py::object {
                auto buf = b.request();
                if (buf.ndim == 1) {
                    auto vb = vector_from_numpy(b);
                    maybe_warn_bit_promotes_to_int32(
                        "matmul",
                        promotion::BinaryOp::MatrixVectorMultiply,
                        a.get_data_type(),
                        vb->get_data_type());
                    auto out = pycauset::matrix_vector_multiply(a, *vb, "");
                    return py::cast(std::shared_ptr<VectorBase>(out.release()));
                }
                if (buf.ndim == 2) {
                    auto mb = matrix_from_numpy(b);
                    maybe_warn_float_underpromotion("matmul", promotion::BinaryOp::Matmul, a, *mb);
                    maybe_warn_bit_promotes_to_int32(
                        "matmul",
                        promotion::BinaryOp::Matmul,
                        a.get_data_type(),
                        mb->get_data_type());
                    maybe_warn_integer_matmul_overflow_risk_preflight(a, *mb);
                    auto out = pycauset::dispatch_matmul(a, *mb, "");
                    return py::cast(std::shared_ptr<MatrixBase>(out.release()));
                }
                throw py::value_error("Unsupported array rank for matmul");
            },
            py::is_operator())
        .def(
            "__matmul__",
            [](const MatrixBase&, const py::object&) -> py::object {
                // Allow other operand types (e.g., internal BlockMatrix) to
                // handle the operation via their __rmatmul__.
                return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            },
            py::is_operator())
        .def("__add__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Add, false); }, py::is_operator())
        .def("__radd__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Add, true); }, py::is_operator())
        .def("__sub__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Sub, false); }, py::is_operator())
        .def("__rsub__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Sub, true); }, py::is_operator())
        .def("__mul__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Mul, false); }, py::is_operator())
        .def("__rmul__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Mul, true); }, py::is_operator())
        .def("__truediv__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Div, false); }, py::is_operator())
        .def("__rtruediv__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Div, true); }, py::is_operator())
        .def(
            "__iadd__",
            [](const std::shared_ptr<MatrixBase>& a, const std::shared_ptr<MatrixBase>& b) {
                maybe_warn_bit_promotes_to_int32(
                    "add",
                    promotion::BinaryOp::Add,
                    a->get_data_type(),
                    b->get_data_type());
                auto out = pycauset::add(*a, *b, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__iadd__",
            [](const std::shared_ptr<MatrixBase>& a, double s) {
                auto out = a->add_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__iadd__",
            [](const std::shared_ptr<MatrixBase>& a, int64_t s) {
                auto out = a->add_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__neg__",
            [](const MatrixBase& a) {
                // F-002: unary minus (wraps for unsigned ints like NumPy; bit/bool raises).
                return std::shared_ptr<MatrixBase>(pycauset::negate(a).release());
            },
            py::is_operator())
        .def(
            "__pos__",
            [](const MatrixBase& a) {
                return std::shared_ptr<MatrixBase>(a.multiply_scalar(static_cast<int64_t>(1)).release());
            },
            py::is_operator());


    // Internal eager elementwise helpers (used by BlockMatrix thunk evaluation).
    m.def(
        "_elementwise_add",
        [](const MatrixBase& a, const MatrixBase& b) {
            maybe_warn_float_underpromotion("add", promotion::BinaryOp::Add, a, b);
            maybe_warn_bit_promotes_to_int32(
                "add",
                promotion::BinaryOp::Add,
                a.get_data_type(),
                b.get_data_type());
            auto out = pycauset::add(a, b, "");
            return std::shared_ptr<MatrixBase>(out.release());
        },
        py::arg("a"),
        py::arg("b"));
    m.def(
        "_elementwise_subtract",
        [](const MatrixBase& a, const MatrixBase& b) {
            maybe_warn_float_underpromotion("subtract", promotion::BinaryOp::Subtract, a, b);
            maybe_warn_bit_promotes_to_int32(
                "subtract",
                promotion::BinaryOp::Subtract,
                a.get_data_type(),
                b.get_data_type());
            auto out = pycauset::subtract(a, b, "");
            return std::shared_ptr<MatrixBase>(out.release());
        },
        py::arg("a"),
        py::arg("b"));
    m.def(
        "_elementwise_multiply",
        [](const MatrixBase& a, const MatrixBase& b) {
            maybe_warn_float_underpromotion("elementwise_multiply", promotion::BinaryOp::ElementwiseMultiply, a, b);
            auto out = pycauset::elementwise_multiply(a, b, "");
            return std::shared_ptr<MatrixBase>(out.release());
        },
        py::arg("a"),
        py::arg("b"));
    m.def(
        "_elementwise_divide",
        [](const MatrixBase& a, const MatrixBase& b) {
            maybe_warn_float_underpromotion("elementwise_divide", promotion::BinaryOp::Divide, a, b);
            auto out = pycauset::elementwise_divide(a, b, "");
            return std::shared_ptr<MatrixBase>(out.release());
        },
        py::arg("a"),
        py::arg("b"));

    // Encourage NumPy to prefer MatrixBase reverse ops over coercion.
    py::object mb_cls = m.attr("MatrixBase");
    mb_cls.attr("__array_priority__") = py::float_(1000.0);

    // Native-side export ceiling so NumPy coercion paths (buffer protocol / native __array__)
    // honor the same policy as pycauset.to_numpy(...).
    m.def(
        "_set_numpy_export_max_bytes",
        [](py::object limit) {
            if (limit.is_none()) {
                pycauset::bindings::g_numpy_export_max_bytes.store(-1);
                return;
            }
            const int64_t v = limit.cast<int64_t>();
            pycauset::bindings::g_numpy_export_max_bytes.store(v);
        },
        py::arg("limit"));
    m.def("_get_numpy_export_max_bytes", []() { return pycauset::bindings::g_numpy_export_max_bytes.load(); });

    // ------------------------------------------------------------------------------------
    // One Python class per structure; the dtype is an attribute (RESTRUCTURE 3.1 item 4).
    // Every method is generic over dtype through pycauset::visit_matrix (DTypeDispatch.hpp);
    // the polymorphic type hook in bindings_common.hpp maps each object to its class.
    // ------------------------------------------------------------------------------------
    matrix_base_class
        .def_property_readonly(
            "dtype",
            [](const MatrixBase& mat) { return std::string(pycauset::dtype_name(mat.get_data_type())); },
            "Element type, e.g. 'float64', 'int32', 'bool' (bit), 'complex_float64'.")
        .def("get", [](const MatrixBase& mat, uint64_t i, uint64_t j) { return element_get(mat, i, j); },
             py::arg("i"), py::arg("j"))
        .def("set", [](MatrixBase& mat, uint64_t i, uint64_t j, const py::object& value) { element_set(mat, i, j, value); },
             py::arg("i"), py::arg("j"), py::arg("value"))
        .def("__getitem__", [](const MatrixBase& mat, const py::object& key) -> py::object {
            return matrixbase_getitem_dispatch(mat, key);
        })
        .def("__setitem__", [](MatrixBase& mat, const py::object& key, const py::object& value) {
            matrixbase_setitem_dispatch(mat, key, value);
        })
        .def("fill", [](MatrixBase& mat, const py::object& value) {
            pycauset::visit_matrix(mat, [&](auto& typed) {
                using M = std::remove_reference_t<decltype(typed)>;
                if constexpr (requires { typed.fill(element_from_py<M>(value)); }) {
                    typed.fill(element_from_py<M>(value));
                } else {
                    throw py::type_error(std::string("fill() is not supported for ") + describe(mat));
                }
            });
        }, py::arg("value"))
        .def("__invert__", [](const MatrixBase& mat_in) -> std::shared_ptr<MatrixBase> {
            // bitwise_not reads raw storage: materialize integer views first (F-042).
            std::unique_ptr<MatrixBase> plain;
            if (pycauset::is_integer_or_bit(mat_in.get_data_type()) && mat_in.get_data_type() != DataType::BIT &&
                mat_in.get_scalar() != std::complex<double>(1.0, 0.0)) {
                plain = pycauset::materialize(mat_in);
            }
            const MatrixBase& mat = plain ? *plain : mat_in;
            return pycauset::visit_matrix(mat, [&](const auto& typed) -> std::shared_ptr<MatrixBase> {
                using E = element_of_t<std::decay_t<decltype(typed)>>;
                if constexpr (std::is_integral_v<E> && requires { typed.bitwise_not(std::string()); }) {
                    return std::shared_ptr<MatrixBase>(typed.bitwise_not("").release());
                } else {
                    throw py::type_error(std::string("~ (bitwise not) is not supported for ") + describe(mat));
                }
            });
        }, py::is_operator())
        .def_property_readonly("is_lazy", [](const MatrixBase& mat) {
            if (auto* c = dynamic_cast<const pycauset::BitClosureMatrix*>(&mat)) {
                if (!c->is_materialized()) return true;
            }
            return mat.get_scalar() != std::complex<double>(1.0, 0.0) || mat.is_transposed() || mat.is_conjugated();
        }, "True when the elements are computed on read (a view or closure) rather than stored.")
        .def("inverse", [](const MatrixBase& mat) { return matrix_inverse(mat); })
        .def("invert", [](const MatrixBase& mat) { return matrix_inverse(mat); })
        .def("matmul", [](const MatrixBase& a, const MatrixBase& b) {
            return std::shared_ptr<MatrixBase>(pycauset::dispatch_matmul(a, b, "").release());
        }, py::arg("other"))
        .def("eig", [](const MatrixBase& mat) {
            auto r = pycauset::eig(mat);
            return py::make_tuple(std::shared_ptr<VectorBase>(r.first.release()), std::shared_ptr<MatrixBase>(r.second.release()));
        })
        .def("eigvals", [](const MatrixBase& mat) { return std::shared_ptr<VectorBase>(pycauset::eigvals(mat).release()); })
        .def("eigh", [](const MatrixBase& mat) {
            auto r = pycauset::eigh(mat);
            return py::make_tuple(std::shared_ptr<VectorBase>(r.first.release()), std::shared_ptr<MatrixBase>(r.second.release()));
        })
        .def("eigvalsh", [](const MatrixBase& mat) { return std::shared_ptr<VectorBase>(pycauset::eigvalsh(mat).release()); });

    // --- DenseMatrix ---------------------------------------------------------------------
    py::class_<pycauset::DenseMatrixBase, MatrixBase, std::shared_ptr<pycauset::DenseMatrixBase>>(
        m, "DenseMatrix", "Dense matrix of any dtype. Prefer pycauset.matrix / zeros / empty to construct.")
        .def(py::init([](uint64_t rows, std::optional<uint64_t> cols, const std::string& dtype) {
                 return make_structure<pycauset::DenseMatrixBase>(
                     rows, cols.value_or(rows), pycauset::Structure::DENSE, dtype);
             }),
             py::arg("rows"), py::arg("cols") = std::nullopt, py::arg("dtype") = "float64")
        .def("set_identity", [](pycauset::DenseMatrixBase& mat) {
            pycauset::visit_matrix(mat, [&](auto& typed) {
                if constexpr (requires { typed.set_identity(); }) {
                    typed.set_identity();
                } else {
                    throw py::type_error(std::string("set_identity() is not supported for ") + describe(mat));
                }
            });
        })
        .def("__and__", [](const MatrixBase& a, const MatrixBase& b) { return bit_binary(a, b, '&'); }, py::is_operator())
        .def("__or__", [](const MatrixBase& a, const MatrixBase& b) { return bit_binary(a, b, '|'); }, py::is_operator())
        .def("__xor__", [](const MatrixBase& a, const MatrixBase& b) { return bit_binary(a, b, '^'); }, py::is_operator())
        .def_static(
            "random",
            [](uint64_t rows, std::optional<uint64_t> cols, double p, std::optional<uint64_t> seed) {
                auto out = DenseBitMatrix::random(rows, cols.value_or(rows), p, "", seed);
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::arg("rows"), py::arg("cols") = std::nullopt, py::arg("p") = 0.5, py::arg("seed") = std::nullopt,
            "Random bit (dtype 'bool') matrix; each entry is 1 with probability p.");

    // --- TriangularMatrix (strictly upper; a causal matrix is TriangularMatrix of dtype 'bool') ---
    py::class_<TriangularMatrixBase, MatrixBase, std::shared_ptr<TriangularMatrixBase>>(
        m, "TriangularMatrix", "Upper-triangular matrix (strict unless has_diagonal). Causal matrices have dtype 'bool'.")
        .def(py::init([](uint64_t n, const std::string& dtype, bool has_diagonal) {
                 const auto dt = pycauset::parse_dtype_name(dtype);
                 return pycauset::visit_dtype(dt, [&](auto tag) -> std::shared_ptr<TriangularMatrixBase> {
                     using T = typename decltype(tag)::type;
                     if constexpr (pycauset::is_complex_element_v<T>) {
                         pycauset::throw_unsupported(pycauset::Structure::TRIANGULAR, dt);
                     } else if constexpr (std::is_same_v<T, bool>) {
                         if (has_diagonal) throw py::value_error("bit triangular matrices are strictly upper triangular");
                         return std::make_shared<TriangularMatrix<bool>>(n, "");
                     } else {
                         return std::make_shared<TriangularMatrix<T>>(n, "", has_diagonal);
                     }
                 });
             }),
             py::arg("n"), py::arg("dtype") = "float64", py::arg("has_diagonal") = false)
        .def_property_readonly("has_diagonal", &TriangularMatrixBase::has_diagonal)
        .def_static(
            "random",
            [](uint64_t n, double p, std::optional<uint64_t> seed) {
                auto out = TriangularBitMatrix::random(n, p, "", seed);
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::arg("n"), py::arg("p") = 0.5, py::arg("seed") = std::nullopt,
            "Random strictly upper-triangular bit (dtype 'bool') matrix; each entry is 1 with probability p.");

    // --- DiagonalMatrix / IdentityMatrix ---------------------------------------------------
    py::class_<pycauset::DiagonalMatrixBase, MatrixBase, std::shared_ptr<pycauset::DiagonalMatrixBase>>(
        m, "DiagonalMatrix", "Diagonal matrix (stores n elements).")
        .def(py::init([](uint64_t n, const std::string& dtype) {
                 return make_structure<pycauset::DiagonalMatrixBase>(n, n, pycauset::Structure::DIAGONAL, dtype);
             }),
             py::arg("n"), py::arg("dtype") = "float64")
        .def("get_diagonal", [](const pycauset::DiagonalMatrixBase& mat, uint64_t i) { return element_get(mat, i, i); },
             py::arg("i"))
        .def("set_diagonal", [](pycauset::DiagonalMatrixBase& mat, uint64_t i, const py::object& v) { element_set(mat, i, i, v); },
             py::arg("i"), py::arg("value"));

    py::class_<pycauset::IdentityMatrixBase, MatrixBase, std::shared_ptr<pycauset::IdentityMatrixBase>>(
        m, "IdentityMatrix", "Identity matrix (no element storage; read-only).")
        .def(py::init([](uint64_t rows, std::optional<uint64_t> cols, const std::string& dtype) {
                 return make_structure<pycauset::IdentityMatrixBase>(
                     rows, cols.value_or(rows), pycauset::Structure::IDENTITY, dtype);
             }),
             py::arg("rows"), py::arg("cols") = std::nullopt, py::arg("dtype") = "float64");

    // --- SymmetricMatrix / AntiSymmetricMatrix (packed upper triangle) ---------------------
    // Both C++ markers are bases of every SymmetricMatrix<T>; Python keeps the two classes
    // unrelated (an antisymmetric matrix is not symmetric), hence MatrixBase as the base.
    py::class_<pycauset::SymmetricMatrixBase, MatrixBase, std::shared_ptr<pycauset::SymmetricMatrixBase>>(
        m, "SymmetricMatrix", "Symmetric matrix, A == A.T (packed upper triangle).")
        .def(py::init([](uint64_t n, const std::string& dtype) {
                 return make_structure<pycauset::SymmetricMatrixBase>(n, n, pycauset::Structure::SYMMETRIC, dtype);
             }),
             py::arg("n"), py::arg("dtype") = "float64")
        .def_static("from_triangular", [](const MatrixBase& source) { return symmetric_from_triangular(source, false); },
                    py::arg("source"), "Symmetric matrix whose upper triangle is `source`'s.");

    py::class_<pycauset::AntiSymmetricMatrixBase, MatrixBase, std::shared_ptr<pycauset::AntiSymmetricMatrixBase>>(
        m, "AntiSymmetricMatrix", "Antisymmetric matrix, A == -A.T (packed strict upper triangle).")
        .def(py::init([](uint64_t n, const std::string& dtype) {
                 return make_structure<pycauset::AntiSymmetricMatrixBase>(n, n, pycauset::Structure::ANTISYMMETRIC, dtype);
             }),
             py::arg("n"), py::arg("dtype") = "float64")
        .def_static("from_triangular", [](const MatrixBase& source) { return symmetric_from_triangular(source, true); },
                    py::arg("source"), "Antisymmetric matrix whose strict upper triangle is `source`'s.");

    // --- Re-open stored data (persistence.load) ---------------------------------------------
    m.def(
        "_open_storage",
        [](const std::string& structure, const std::string& dtype, uint64_t rows, uint64_t cols,
           const std::string& path, size_t offset, uint64_t seed, std::complex<double> scalar,
           bool is_transposed) -> py::object {
            return open_storage(pycauset::parse_structure(structure), pycauset::parse_dtype_name(dtype), rows, cols,
                                path, offset, seed, scalar, is_transposed);
        },
        py::arg("structure"), py::arg("dtype"), py::arg("rows"), py::arg("cols"), py::arg("path"),
        py::arg("offset"), py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"),
        "Internal: map an existing data file as a matrix or vector (storage shape; the view flags are applied).");
    m.def("_materialize_for_save", [](const std::shared_ptr<MatrixBase>& mat) -> std::shared_ptr<MatrixBase> {
        // Closure views read another object's storage layout; files store the packed result.
        if (auto* c = dynamic_cast<const pycauset::BitClosureMatrix*>(mat.get())) {
            return std::shared_ptr<MatrixBase>(c->materialized_copy().release());
        }
        return mat;
    }, py::arg("matrix"));
    m.def("materialize", [](const std::shared_ptr<MatrixBase>& mat) -> std::shared_ptr<MatrixBase> {
        if (auto* c = dynamic_cast<const pycauset::BitClosureMatrix*>(mat.get())) {
            return std::shared_ptr<MatrixBase>(c->materialized_copy().release());
        }
        if (mat->get_scalar() != std::complex<double>(1.0, 0.0) || mat->is_transposed() || mat->is_conjugated()) {
            return std::shared_ptr<MatrixBase>(pycauset::materialize(*mat).release());
        }
        return mat;
    }, py::arg("matrix"), "The matrix with its elements stored (views and closures evaluated).");
    m.def("_structure_of", [](const py::object& obj) -> std::string {
        if (py::isinstance<MatrixBase>(obj)) return pycauset::structure_name(obj.cast<const MatrixBase&>().get_structure());
        if (py::isinstance<VectorBase>(obj)) return pycauset::structure_name(obj.cast<const VectorBase&>().get_structure());
        throw py::type_error("not a pycauset matrix or vector");
    }, py::arg("obj"), "Internal: the storage structure name ('dense', 'triangular', ...).");

    m.def("asarray", [](const py::array& array) -> py::object {
        auto buf = array.request();
        if (buf.ndim == 1) {
            if (is_numpy_float16(array)) {
                auto b = array.request();
                require_1d(b);
                uint64_t n = static_cast<uint64_t>(b.shape[0]);
                auto result = std::make_shared<DenseVector<pycauset::float16_t>>(n);
                const uint16_t* src_ptr = static_cast<const uint16_t*>(b.ptr);
                pycauset::float16_t* dst_ptr = result->data();
                if (b.strides[0] == static_cast<py::ssize_t>(sizeof(uint16_t))) {
                    std::memcpy(static_cast<void*>(dst_ptr), src_ptr, n * sizeof(uint16_t));
                } else {
                    const ptrdiff_t stride0 = static_cast<ptrdiff_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                    for (uint64_t i = 0; i < n; ++i) {
                        const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0;
                        dst_ptr[i] = pycauset::float16_t(static_cast<uint16_t>(src_ptr[idx]));
                    }
                }
                return py::cast(result);
            }
            if (py::isinstance<py::array_t<double>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<double>(array.cast<py::array_t<double>>()));
            }
            if (py::isinstance<py::array_t<float>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<float>(array.cast<py::array_t<float>>()));  // dtype kept (F-056)
            }
            if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<std::complex<float>>(array.cast<py::array_t<std::complex<float>>>()));
            }
            if (py::isinstance<py::array_t<std::complex<double>>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<std::complex<double>>(array.cast<py::array_t<std::complex<double>>>()));
            }
            if (py::isinstance<py::array_t<int32_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<int32_t>(array.cast<py::array_t<int32_t>>()));
            }
            if (py::isinstance<py::array_t<int64_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<int64_t>(array.cast<py::array_t<int64_t>>()));
            }
            if (py::isinstance<py::array_t<int8_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<int8_t>(array.cast<py::array_t<int8_t>>()));
            }
            if (py::isinstance<py::array_t<int16_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<int16_t>(array.cast<py::array_t<int16_t>>()));
            }
            if (py::isinstance<py::array_t<uint8_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<uint8_t>(array.cast<py::array_t<uint8_t>>()));
            }
            if (py::isinstance<py::array_t<uint16_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<uint16_t>(array.cast<py::array_t<uint16_t>>()));
            }
            if (py::isinstance<py::array_t<uint32_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<uint32_t>(array.cast<py::array_t<uint32_t>>()));
            }
            if (py::isinstance<py::array_t<uint64_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<uint64_t>(array.cast<py::array_t<uint64_t>>()));
            }
            if (py::isinstance<py::array_t<bool>>(array)) {
                auto tmp = array.cast<py::array_t<bool>>();
                auto b = tmp.request();
                require_1d(b);
                uint64_t n = static_cast<uint64_t>(b.shape[0]);
                auto result = std::make_shared<DenseVector<bool>>(n);
                const bool* src_ptr = static_cast<const bool*>(b.ptr);
                const ptrdiff_t stride0 = static_cast<ptrdiff_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
                for (uint64_t i = 0; i < n; ++i) {
                    const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0;
                    result->set(i, src_ptr[idx]);
                }
                return py::cast(result);
            }
            throw py::type_error("Unsupported dtype for 1D asarray");
        }

        if (buf.ndim == 2) {
            if (is_numpy_float16(array)) {
                return py::cast(matrix_from_numpy(array));
            }
            if (py::isinstance<py::array_t<double>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<double>(array.cast<py::array_t<double>>()));
            }
            if (py::isinstance<py::array_t<float>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<float>(array.cast<py::array_t<float>>()));
            }
            if (py::isinstance<py::array_t<int32_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<int32_t>(array.cast<py::array_t<int32_t>>()));
            }
            if (py::isinstance<py::array_t<int64_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<int64_t>(array.cast<py::array_t<int64_t>>()));
            }
            if (py::isinstance<py::array_t<int8_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<int8_t>(array.cast<py::array_t<int8_t>>()));
            }
            if (py::isinstance<py::array_t<int16_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<int16_t>(array.cast<py::array_t<int16_t>>()));
            }
            if (py::isinstance<py::array_t<uint8_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<uint8_t>(array.cast<py::array_t<uint8_t>>()));
            }
            if (py::isinstance<py::array_t<uint16_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<uint16_t>(array.cast<py::array_t<uint16_t>>()));
            }
            if (py::isinstance<py::array_t<uint32_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<uint32_t>(array.cast<py::array_t<uint32_t>>()));
            }
            if (py::isinstance<py::array_t<uint64_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<uint64_t>(array.cast<py::array_t<uint64_t>>()));
            }
            if (py::isinstance<py::array_t<bool>>(array)) {
                auto tmp = array.cast<py::array_t<bool>>();
                auto b = tmp.request();
                require_2d(b);
                uint64_t rows = static_cast<uint64_t>(b.shape[0]);
                uint64_t cols = static_cast<uint64_t>(b.shape[1]);
                auto result = std::make_shared<DenseBitMatrix>(rows, cols);
                const bool* src_ptr = static_cast<const bool*>(b.ptr);
                const ptrdiff_t stride0 = static_cast<ptrdiff_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
                const ptrdiff_t stride1 = static_cast<ptrdiff_t>(b.strides[1] / static_cast<py::ssize_t>(sizeof(bool)));
                
                uint64_t* dst_data = result->data();
                uint64_t dst_stride_bytes = result->stride_bytes();
                uint64_t dst_stride_words = dst_stride_bytes / 8;
                
                if (dst_data) {
                    for (uint64_t i = 0; i < rows; ++i) {
                        uint64_t* row_ptr = dst_data + i * dst_stride_words;
                        const bool* src_row = src_ptr + static_cast<ptrdiff_t>(i) * stride0;
                        
                        for (uint64_t j = 0; j < cols; j += 64) {
                            uint64_t word = 0;
                            uint64_t limit = std::min(cols - j, (uint64_t)64);
                            
                            if (stride1 == 1) {
                                const bool* src_chunk = src_row + j;
                                uint64_t k = 0;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                                __m128i zero = _mm_setzero_si128();
                                while (k + 16 <= limit) {
                                    __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_chunk + k));
                                    __m128i cmp = _mm_cmpeq_epi8(chunk, zero);
                                    int mask = _mm_movemask_epi8(cmp);
                                    uint64_t bits = static_cast<uint16_t>(~mask);
                                    word |= (bits << k);
                                    k += 16;
                                }
#endif
                                for (; k < limit; ++k) {
                                    if (src_chunk[k]) {
                                        word |= (1ULL << k);
                                    }
                                }
                            } else {
                                for (uint64_t k = 0; k < limit; ++k) {
                                    if (src_row[(j + k) * stride1]) {
                                        word |= (1ULL << k);
                                    }
                                }
                            }
                            row_ptr[j / 64] = word;
                        }
                    }
                    return py::cast(result);
                }

                for (uint64_t i = 0; i < rows; ++i) {
                    for (uint64_t j = 0; j < cols; ++j) {
                        const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0 + static_cast<ptrdiff_t>(j) * stride1;
                        result->set(i, j, src_ptr[idx]);
                    }
                }
                return py::cast(result);
            }
            if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<std::complex<float>>(array.cast<py::array_t<std::complex<float>>>()));
            }
            if (py::isinstance<py::array_t<std::complex<double>>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<std::complex<double>>(array.cast<py::array_t<std::complex<double>>>()));
            }
            throw py::type_error("Unsupported dtype for 2D asarray");
        }

        throw py::value_error("asarray only supports 1D or 2D arrays");
    });


    m.def(
        "matmul",
        [](std::shared_ptr<MatrixBase> a, std::shared_ptr<MatrixBase> b) {
            return translate_invalid_argument([&]() {
                maybe_warn_float_underpromotion("matmul", promotion::BinaryOp::Matmul, *a, *b);
                maybe_warn_bit_promotes_to_int32(
                    "matmul",
                    promotion::BinaryOp::Matmul,
                    a->get_data_type(),
                    b->get_data_type());
                maybe_warn_integer_matmul_overflow_risk_preflight(*a, *b);
                auto out = pycauset::dispatch_matmul(*a, *b, "");
                return std::shared_ptr<MatrixBase>(out.release());
            });
        },
        py::arg("a"),
        py::arg("b"));

    m.def(
        "cholesky",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::cholesky(*a);
                return std::shared_ptr<MatrixBase>(out.release());
            });
        },
        py::arg("a"));

    m.def(
        "eig",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eig(*a);
                return std::make_pair(
                    std::shared_ptr<VectorBase>(out.first.release()),
                    std::shared_ptr<MatrixBase>(out.second.release())
                );
            });
        },
        py::arg("a"));

    m.def(
        "eigvals",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigvals(*a);
                return std::shared_ptr<VectorBase>(out.release());
            });
        },
        py::arg("a"));

    m.def(
        "eigh",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigh(*a);
                return std::make_pair(
                    std::shared_ptr<VectorBase>(out.first.release()),
                    std::shared_ptr<MatrixBase>(out.second.release())
                );
            });
        },
        py::arg("a"));

    m.def(
        "eigvalsh",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigvalsh(*a);
                return std::shared_ptr<VectorBase>(out.release());
            });
        },
        py::arg("a"));

    m.def(
        "eigvals_arnoldi",
        [](std::shared_ptr<MatrixBase> a, int k, int m, double tol) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigvals_arnoldi(*a, k, m, tol);
                return std::shared_ptr<VectorBase>(out.release());
            });
        },
        py::arg("a"),
        py::arg("k"),
        py::arg("m"),
        py::arg("tol") = 1e-6);

    m.def(
        "eigvals_skew",
        [](std::shared_ptr<MatrixBase> a, int k) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigvals_skew(*a, k);
                return std::shared_ptr<VectorBase>(out.release());
            });
        },
        py::arg("a"),
        py::arg("k"));

    m.def(
        "eig_skew",
        [](std::shared_ptr<MatrixBase> a, int k) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eig_skew(*a, k);
                return std::make_pair(
                    std::shared_ptr<VectorBase>(out.first.release()),
                    std::shared_ptr<MatrixBase>(out.second.release())
                );
            });
        },
        py::arg("a"),
        py::arg("k"));

    m.def(
        "compute_k_matrix",
        [](std::shared_ptr<MatrixBase> C_any,
           double a,
           int num_threads) {
            auto* C = dynamic_cast<TriangularBitMatrix*>(C_any.get());
            if (C == nullptr) {
                throw py::type_error("compute_k_matrix expects a causal matrix (TriangularMatrix of dtype 'bool')");
            }
            auto out = ComputeContext::instance().get_device()->compute_k_matrix(*C, a, "", num_threads);
            return std::shared_ptr<TriangularMatrix<double>>(out.release());
        },
        py::arg("C"),
        py::arg("a"),
        py::arg("num_threads") = 0);

}
