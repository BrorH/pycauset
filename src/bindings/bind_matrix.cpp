#include "bindings_common.hpp"

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

#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/VectorBase.hpp"

#include "numpy_export_guard.hpp"

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

template <typename ScalarT>
inline const char* dense_matrix_buffer_format() {
    static const std::string fmt = py::format_descriptor<ScalarT>::format();
    return fmt.c_str();
}

template <>
inline const char* dense_matrix_buffer_format<float16_t>() {
    return "e"; // PEP 3118: float16
}

template <typename ScalarT>
int dense_matrix_getbuffer(PyObject* obj, Py_buffer* view, int flags) {
    if (view == nullptr) {
        PyErr_SetString(PyExc_BufferError, "NULL view in getbuffer");
        return -1;
    }

    DenseMatrix<ScalarT>* m = nullptr;
    try {
        m = &py::handle(obj).cast<DenseMatrix<ScalarT>&>();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }

    try {
        pycauset::bindings::ensure_numpy_export_allowed(static_cast<const MatrixBase&>(*m));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }

    // Buffer exports expose raw storage; they are only correct for unscaled, unconjugated views.
    // For scalar/conjugated matrices, fall back to __array__/materialization paths.
    if (m->get_scalar() != std::complex<double>(1.0, 0.0) || m->is_conjugated()) {
        PyErr_SetString(PyExc_BufferError, "Buffer export requires scalar==1 and not conjugated");
        return -1;
    }

    Py_ssize_t rows_ss = 0;
    Py_ssize_t cols_ss = 0;
    if (!u64_to_py_ssize(m->rows(), &rows_ss) || !u64_to_py_ssize(m->cols(), &cols_ss)) {
        PyErr_SetString(PyExc_OverflowError, "Matrix too large for Python buffer protocol");
        return -1;
    }

    uint64_t stride0_u64 = 0;
    uint64_t stride1_u64 = 0;
    if (pycauset::bindings::mul_overflow_u64(sizeof(ScalarT), m->base_cols(), &stride0_u64)) {
        PyErr_SetString(PyExc_OverflowError, "Stride overflow in Python buffer protocol");
        return -1;
    }
    stride1_u64 = sizeof(ScalarT);
    if (m->is_transposed()) {
        std::swap(stride0_u64, stride1_u64);
    }

    if (stride0_u64 > static_cast<uint64_t>(PY_SSIZE_T_MAX) || stride1_u64 > static_cast<uint64_t>(PY_SSIZE_T_MAX)) {
        PyErr_SetString(PyExc_OverflowError, "Stride too large for Python buffer protocol");
        return -1;
    }

    const uint64_t rows_u64 = m->rows();
    const uint64_t cols_u64 = m->cols();
    uint64_t len_u64 = 0;
    if (rows_u64 == 0 || cols_u64 == 0) {
        len_u64 = 0;
    } else {
        uint64_t off0 = 0;
        uint64_t off1 = 0;
        uint64_t tmp = 0;
        if (pycauset::bindings::mul_overflow_u64(rows_u64 - 1, stride0_u64, &off0) ||
            pycauset::bindings::mul_overflow_u64(cols_u64 - 1, stride1_u64, &off1) ||
            add_overflow_u64(off0, off1, &tmp) ||
            add_overflow_u64(tmp, sizeof(ScalarT), &len_u64)) {
            PyErr_SetString(PyExc_OverflowError, "Buffer size overflow in Python buffer protocol");
            return -1;
        }
    }

    if (len_u64 > static_cast<uint64_t>(PY_SSIZE_T_MAX)) {
        PyErr_SetString(PyExc_OverflowError, "Buffer too large for Python buffer protocol");
        return -1;
    }

    const auto len_ss = static_cast<Py_ssize_t>(len_u64);
    if (PyBuffer_FillInfo(view, obj, static_cast<void*>(m->data()), len_ss, /*readonly=*/0, flags) != 0) {
        return -1;
    }

    auto* state = new DenseMatrix2DBufferState();
    state->shape[0] = rows_ss;
    state->shape[1] = cols_ss;
    state->strides[0] = static_cast<Py_ssize_t>(stride0_u64);
    state->strides[1] = static_cast<Py_ssize_t>(stride1_u64);

    view->internal = state;
    view->itemsize = static_cast<Py_ssize_t>(sizeof(ScalarT));
    view->ndim = 2;
    view->suboffsets = nullptr;

    view->format = (flags & PyBUF_FORMAT) ? const_cast<char*>(dense_matrix_buffer_format<ScalarT>()) : nullptr;
    view->shape = (flags & PyBUF_ND) ? state->shape : nullptr;
    view->strides = (flags & PyBUF_STRIDES) ? state->strides : nullptr;

    return 0;
}

template <typename ScalarT>
void dense_matrix_releasebuffer(PyObject* /*obj*/, Py_buffer* view) {
    auto* state = reinterpret_cast<DenseMatrix2DBufferState*>(view->internal);
    delete state;
    view->internal = nullptr;
}

template <typename ScalarT>
void install_dense_matrix_buffer(py::handle type_handle) {
    auto* type = reinterpret_cast<PyTypeObject*>(type_handle.ptr());
    static PyBufferProcs procs{dense_matrix_getbuffer<ScalarT>, dense_matrix_releasebuffer<ScalarT>};
    type->tp_as_buffer = &procs;
#ifdef Py_TPFLAGS_HAVE_NEWBUFFER
    type->tp_flags |= Py_TPFLAGS_HAVE_NEWBUFFER;
#endif
    PyType_Modified(type);
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

inline void maybe_warn_integer_matmul_accumulator_widen(const MatrixBase& a, const MatrixBase& b) {
    if (!is_dense_int32_matrix(a) || !is_dense_int32_matrix(b)) {
        return;
    }

    // Deterministic warning: this operation uses an int64 accumulator to avoid intermediate overflow,
    // but the output storage remains int32 and will throw on overflow.
    std::string message;
    message.reserve(256);
    message += "pycauset matmul: using int64 accumulator for int32 @ int32 to avoid intermediate overflow; ";
    message += "output stored as int32 and will raise on overflow.";
    bindings_warn::warn_once_with_category(
        "pycauset.matmul.accum_widen.int32",
        message,
        "PyCausetDTypeWarning",
        /*stacklevel=*/3);
}

inline void maybe_warn_integer_matmul_overflow_risk_preflight(const MatrixBase& a, const MatrixBase& b) {
    // Heuristic risk check (cheap): sample a few entries to estimate max-abs,
    // then apply a conservative reduction bound.
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
    message += "pycauset matmul preflight: int32 @ int32 may overflow int32 output (heuristic bound). ";
    message += "m=" + std::to_string(m);
    message += ", k=" + std::to_string(k);
    message += ", n=" + std::to_string(n);
    message += ", sampled max|A|=" + std::to_string(max_abs_a);
    message += ", sampled max|B|=" + std::to_string(max_abs_b);
    message += ", bound≈" + std::to_string(bound);
    message += ". PyCauset will raise on overflow.";
    bindings_warn::warn_once_with_category(
        "pycauset.matmul.overflow_risk_preflight.int32",
        message,
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

    std::string key;
    key.reserve(64);
    key += "pycauset.";
    key += op_name;
    key += ".float_underpromotion.f32_f64";

    std::string message;
    message.reserve(256);
    message += "pycauset ";
    message += op_name;
    message += ": mixed float32/float64 underpromotes to float32 (compute and storage in float32). ";
    message += "Cast inputs to float64 or use pycauset.precision_mode('highest') to promote.";

    bindings_warn::warn_once_with_category(key, message, "PyCausetDTypeWarning", /*stacklevel=*/3);
}

inline void maybe_warn_bit_promotes_to_int32(const char* op_name, promotion::BinaryOp op, DataType da, DataType db) {
    if (!(da == DataType::BIT && db == DataType::BIT)) {
        return;
    }

    const auto decision = promotion::resolve(op, da, db);
    if (decision.result_dtype != DataType::INT32) {
        return;
    }

    std::string key;
    key.reserve(96);
    key += "pycauset.";
    key += op_name;
    key += ".bit_promotes_to_int32";

    std::string message;
    message.reserve(256);
    message += "pycauset ";
    message += op_name;
    message += ": bit operands promote to int32 because the result can exceed {0,1} (mathematically unavoidable). ";
    message += "Even though int16 exists, bit-bit reductions commonly exceed int16; result is int32.";

    bindings_warn::warn_once_with_category(key, message, "PyCausetDTypeWarning", /*stacklevel=*/3);
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

template <typename T>
std::shared_ptr<DenseMatrix<T>> dense_matrix_from_numpy_2d(const py::array_t<T>& array) {
    auto buf = array.request();
    require_2d(buf);

    uint64_t rows = static_cast<uint64_t>(buf.shape[0]);
    uint64_t cols = static_cast<uint64_t>(buf.shape[1]);
    auto result = std::make_shared<DenseMatrix<T>>(rows, cols);

    const T* src_ptr = static_cast<const T*>(buf.ptr);
    T* dst_ptr = result->data();

    const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
    const ptrdiff_t stride1 = static_cast<ptrdiff_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(T)));

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

inline std::shared_ptr<DenseVector<double>> dense_vector_from_numpy_1d_float32_promote_to_float64(const py::array_t<float>& array) {
    auto buf = array.request();
    require_1d(buf);
    const uint64_t n = static_cast<uint64_t>(buf.shape[0]);

    auto result = std::make_shared<DenseVector<double>>(n);
    const float* src_ptr = static_cast<const float*>(buf.ptr);
    double* dst_ptr = result->data();

    const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(float)));

    // Contiguous float32 -> float64 is the hot path; parallelize only when it's worth it.
    if (stride0 == 1) {
        const Eigen::Index n_e = static_cast<Eigen::Index>(n);
        Eigen::Map<const Eigen::ArrayXf> src(src_ptr, n_e);
        Eigen::Map<Eigen::ArrayXd> dst(dst_ptr, n_e);
        dst = src.cast<double>();
        return result;
    }

    for (uint64_t i = 0; i < n; ++i) {
        const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0;
        dst_ptr[i] = static_cast<double>(src_ptr[idx]);
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
        // Promote float32 vectors to float64 vectors for now.
        return dense_vector_from_numpy_1d_float32_promote_to_float64(array.cast<py::array_t<float>>());
    }
    throw py::type_error("Unsupported NumPy dtype for vector conversion");
}

inline std::shared_ptr<MatrixBase> matrix_from_numpy(const py::array& array) {
    if (is_numpy_float16(array)) {
        auto buf = array.request();
        require_2d(buf);

        uint64_t rows = static_cast<uint64_t>(buf.shape[0]);
        uint64_t cols = static_cast<uint64_t>(buf.shape[1]);
        auto result = std::make_shared<DenseMatrix<float16_t>>(rows, cols);

        const uint16_t* src_ptr = static_cast<const uint16_t*>(buf.ptr);
        float16_t* dst_ptr = result->data();

        if (buf.strides[1] == static_cast<py::ssize_t>(sizeof(uint16_t)) &&
            buf.strides[0] == static_cast<py::ssize_t>(cols * sizeof(uint16_t))) {
            std::memcpy(dst_ptr, src_ptr, rows * cols * sizeof(uint16_t));
        } else {
            const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
            const ptrdiff_t stride1 = static_cast<ptrdiff_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(uint16_t)));
            for (uint64_t i = 0; i < rows; ++i) {
                for (uint64_t j = 0; j < cols; ++j) {
                    const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0 + static_cast<ptrdiff_t>(j) * stride1;
                    dst_ptr[i * cols + j] = float16_t(static_cast<uint16_t>(src_ptr[idx]));
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
        auto result = std::make_shared<DenseMatrix<float16_t>>(1, cols);

        const uint16_t* src_ptr = static_cast<const uint16_t*>(buf.ptr);
        float16_t* dst_ptr = result->data();

        if (buf.strides[0] == static_cast<py::ssize_t>(sizeof(uint16_t))) {
            std::memcpy(dst_ptr, src_ptr, cols * sizeof(uint16_t));
        } else {
            const ptrdiff_t stride0 = static_cast<ptrdiff_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
            for (uint64_t j = 0; j < cols; ++j) {
                const ptrdiff_t idx = static_cast<ptrdiff_t>(j) * stride0;
                dst_ptr[j] = float16_t(static_cast<uint16_t>(src_ptr[idx]));
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

inline bool coerce_bit_value(const py::handle& value);

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

    const auto src_name = py::str(src_dtype).cast<std::string>();
    const auto tgt_name = py::str(target_dtype).cast<std::string>();
    std::string msg = "pycauset assignment: casting RHS from ";
    msg += src_name;
    msg += " to ";
    msg += tgt_name;
    bindings_warn::warn_once_with_category(
        "pycauset.assign.cast.",
        msg,
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
            bindings_warn::warn_once_with_category(
                "pycauset.assign.overflow_risk.",
                risk,
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

inline bool coerce_bit_value(const py::handle& value) {
    if (py::isinstance<py::bool_>(value)) {
        return value.cast<bool>();
    }
    if (py::isinstance<py::int_>(value)) {
        auto v = value.cast<long long>();
        if (v == 0) return false;
        if (v == 1) return true;
        throw py::type_error("Bit matrices only accept 0/1/False/True");
    }
    if (py::isinstance<py::float_>(value)) {
        double v = value.cast<double>();
        if (v == 0.0) return false;
        if (v == 1.0) return true;
        throw py::type_error("Bit matrices only accept 0/1/False/True");
    }
    throw py::type_error("Bit matrices only accept 0/1/False/True");
}

inline py::object matrixbase_getitem_dispatch(const MatrixBase& mat, const py::object& key) {
    auto parsed = parse_matrix_subscript(key, mat.rows(), mat.cols());
    if (parsed.rows.is_scalar && parsed.cols.is_scalar) {
        const DataType dt = mat.get_data_type();
        if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
            return py::cast(mat.get_element_as_complex(parsed.rows.index, parsed.cols.index));
        }
        return py::float_(mat.get_element_as_double(parsed.rows.index, parsed.cols.index));
    }

    // Slicing: supported for dense matrix types; others raise for now.
    if (auto* md = dynamic_cast<const DenseMatrix<double>*>(&mat)) {
        return dense_getitem(*md, key);
    }
    if (auto* mf = dynamic_cast<const DenseMatrix<float>*>(&mat)) {
        return dense_getitem(*mf, key);
    }
    if (auto* mf16 = dynamic_cast<const DenseMatrix<float16_t>*>(&mat)) {
        return dense_getitem(*mf16, key);
    }
    if (auto* mi32 = dynamic_cast<const DenseMatrix<int32_t>*>(&mat)) {
        return dense_getitem(*mi32, key);
    }
    if (auto* mi16 = dynamic_cast<const DenseMatrix<int16_t>*>(&mat)) {
        return dense_getitem(*mi16, key);
    }
    if (auto* mi8 = dynamic_cast<const DenseMatrix<int8_t>*>(&mat)) {
        return dense_getitem(*mi8, key);
    }
    if (auto* mi64 = dynamic_cast<const DenseMatrix<int64_t>*>(&mat)) {
        return dense_getitem(*mi64, key);
    }
    if (auto* mu8 = dynamic_cast<const DenseMatrix<uint8_t>*>(&mat)) {
        return dense_getitem(*mu8, key);
    }
    if (auto* mu16 = dynamic_cast<const DenseMatrix<uint16_t>*>(&mat)) {
        return dense_getitem(*mu16, key);
    }
    if (auto* mu32 = dynamic_cast<const DenseMatrix<uint32_t>*>(&mat)) {
        return dense_getitem(*mu32, key);
    }
    if (auto* mu64 = dynamic_cast<const DenseMatrix<uint64_t>*>(&mat)) {
        return dense_getitem(*mu64, key);
    }
    if (auto* mb = dynamic_cast<const DenseMatrix<bool>*>(&mat)) {
        return dense_getitem(*mb, key);
    }
    if (auto* mcf32 = dynamic_cast<const DenseMatrix<std::complex<float>>*>(&mat)) {
        return dense_getitem(*mcf32, key);
    }
    if (auto* mcf64 = dynamic_cast<const DenseMatrix<std::complex<double>>*>(&mat)) {
        return dense_getitem(*mcf64, key);
    }

    PyErr_SetString(PyExc_NotImplementedError, "Slicing not supported for this matrix type yet");
    throw py::error_already_set();
}

template <typename Fn>
inline auto translate_invalid_argument(Fn&& fn) {
    try {
        return fn();
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
}

} // namespace

void bind_matrix_classes(py::module_& m) {
    py::class_<MatrixBase, std::shared_ptr<MatrixBase>>(m, "MatrixBase", py::dynamic_attr())
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
                    maybe_warn_integer_matmul_accumulator_widen(a, *b);
                    maybe_warn_integer_matmul_overflow_risk_preflight(a, *b);
                    auto out = pycauset::dispatch_matmul(a, *b, "");
                    return std::shared_ptr<MatrixBase>(out.release());
                });
            },
            py::arg("other"))
        .def("get_element_as_double", &MatrixBase::get_element_as_double)
        .def("get_element_as_complex", &MatrixBase::get_element_as_complex)
        .def("get", [](const MatrixBase& mat, uint64_t i, uint64_t j) -> py::object {
            const DataType dt = mat.get_data_type();
            if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
                return py::cast(mat.get_element_as_complex(i, j));
            }
            return py::float_(mat.get_element_as_double(i, j));
        })
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

                if (auto* mf16 = dynamic_cast<const DenseMatrix<float16_t>*>(&mat)) {
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
                    auto r = out.mutable_unchecked<2>();
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
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mi32->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mi64 = dynamic_cast<const DenseMatrix<int64_t>*>(&mat)) {
                    py::array_t<int64_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mi64->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mi8 = dynamic_cast<const DenseMatrix<int8_t>*>(&mat)) {
                    py::array_t<int8_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mi8->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mi16 = dynamic_cast<const DenseMatrix<int16_t>*>(&mat)) {
                    py::array_t<int16_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mi16->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mu8 = dynamic_cast<const DenseMatrix<uint8_t>*>(&mat)) {
                    py::array_t<uint8_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mu8->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mu16 = dynamic_cast<const DenseMatrix<uint16_t>*>(&mat)) {
                    py::array_t<uint16_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mu16->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mu32 = dynamic_cast<const DenseMatrix<uint32_t>*>(&mat)) {
                    py::array_t<uint32_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mu32->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mu64 = dynamic_cast<const DenseMatrix<uint64_t>*>(&mat)) {
                    py::array_t<uint64_t> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mu64->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mb = dynamic_cast<const DenseBitMatrix*>(&mat)) {
                    py::array_t<bool> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mb->get(i, j);
                        }
                    }
                    return out;
                }
                if (auto* mtb = dynamic_cast<const TriangularBitMatrix*>(&mat)) {
                    py::array_t<bool> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            r(i, j) = mtb->get(i, j);
                        }
                    }
                    return out;
                }

                // Default: float64
                py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
                auto r = out.mutable_unchecked<2>();
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
        .def("__getitem__", [](const MatrixBase& mat, const py::object& key) -> py::object {
            return matrixbase_getitem_dispatch(mat, key);
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
                    maybe_warn_integer_matmul_accumulator_widen(a, *b);
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
                    maybe_warn_integer_matmul_accumulator_widen(a, *mb);
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
        .def(
            "__add__",
            [](const MatrixBase& a, const std::shared_ptr<MatrixBase>& b) {
                maybe_warn_float_underpromotion("add", promotion::BinaryOp::Add, a, *b);
                maybe_warn_bit_promotes_to_int32(
                    "add",
                    promotion::BinaryOp::Add,
                    a.get_data_type(),
                    b->get_data_type());
                auto out = pycauset::add(a, *b, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__add__",
            [](const MatrixBase& a, const py::array& b) {
                auto mb = elementwise_operand_matrix_from_numpy(b);
                maybe_warn_float_underpromotion("add", promotion::BinaryOp::Add, a, *mb);
                maybe_warn_bit_promotes_to_int32(
                    "add",
                    promotion::BinaryOp::Add,
                    a.get_data_type(),
                    mb->get_data_type());
                auto out = pycauset::add(a, *mb, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__add__",
            [](const MatrixBase& a, double s) {
                auto out = a.add_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__add__",
            [](const MatrixBase& a, int64_t s) {
                auto out = a.add_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__add__",
            [](const MatrixBase&, const py::object&) -> py::object {
                // Allow other operand types (e.g., internal BlockMatrix) to
                // handle the operation via their __radd__.
                return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const MatrixBase& a, double s) {
                auto out = a.add_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const MatrixBase& a, int64_t s) {
                auto out = a.add_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const MatrixBase& a, const py::array& b) {
                auto mb = elementwise_operand_matrix_from_numpy(b);
                maybe_warn_float_underpromotion("add", promotion::BinaryOp::Add, *mb, a);
                maybe_warn_bit_promotes_to_int32(
                    "add",
                    promotion::BinaryOp::Add,
                    mb->get_data_type(),
                    a.get_data_type());
                auto out = pycauset::add(*mb, a, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
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
            "__sub__",
            [](const MatrixBase& a, const std::shared_ptr<MatrixBase>& b) {
                maybe_warn_float_underpromotion("subtract", promotion::BinaryOp::Subtract, a, *b);
                maybe_warn_bit_promotes_to_int32(
                    "subtract",
                    promotion::BinaryOp::Subtract,
                    a.get_data_type(),
                    b->get_data_type());
                auto out = pycauset::subtract(a, *b, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__sub__",
            [](const MatrixBase& a, const py::array& b) {
                auto mb = elementwise_operand_matrix_from_numpy(b);
                maybe_warn_float_underpromotion("subtract", promotion::BinaryOp::Subtract, a, *mb);
                maybe_warn_bit_promotes_to_int32(
                    "subtract",
                    promotion::BinaryOp::Subtract,
                    a.get_data_type(),
                    mb->get_data_type());
                auto out = pycauset::subtract(a, *mb, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__sub__",
            [](const MatrixBase&, const py::object&) -> py::object {
                // Allow other operand types (e.g., internal BlockMatrix) to
                // handle the operation via their __rsub__.
                return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            },
            py::is_operator())
        .def(
            "__rsub__",
            [](const MatrixBase& a, const py::array& b) {
                auto mb = elementwise_operand_matrix_from_numpy(b);
                maybe_warn_float_underpromotion("subtract", promotion::BinaryOp::Subtract, *mb, a);
                maybe_warn_bit_promotes_to_int32(
                    "subtract",
                    promotion::BinaryOp::Subtract,
                    mb->get_data_type(),
                    a.get_data_type());
                auto out = pycauset::subtract(*mb, a, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const MatrixBase& a, const std::shared_ptr<MatrixBase>& b) {
                maybe_warn_float_underpromotion("elementwise_divide", promotion::BinaryOp::Divide, a, *b);
                auto out = pycauset::elementwise_divide(a, *b, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const MatrixBase& a, const py::array& b) {
                auto mb = elementwise_operand_matrix_from_numpy(b);
                maybe_warn_float_underpromotion("elementwise_divide", promotion::BinaryOp::Divide, a, *mb);
                auto out = pycauset::elementwise_divide(a, *mb, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const MatrixBase& a, double s) {
                auto out = a.multiply_scalar(1.0 / s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const MatrixBase& a, int64_t s) {
                auto out = a.multiply_scalar(1.0 / static_cast<double>(s), "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const MatrixBase& a, std::complex<double> s) {
                auto out = a.multiply_scalar(1.0 / s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__rtruediv__",
            [](const MatrixBase& a, const py::array& b) {
                auto mb = elementwise_operand_matrix_from_numpy(b);
                maybe_warn_float_underpromotion("elementwise_divide", promotion::BinaryOp::Divide, *mb, a);
                auto out = pycauset::elementwise_divide(*mb, a, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](const MatrixBase&, const py::object&) -> py::object {
                // Allow other operand types (e.g., internal BlockMatrix) to
                // handle the operation via their __rtruediv__.
                return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const MatrixBase& a, const std::shared_ptr<MatrixBase>& b) {
                maybe_warn_float_underpromotion("elementwise_multiply", promotion::BinaryOp::ElementwiseMultiply, a, *b);
                auto out = pycauset::elementwise_multiply(a, *b, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const MatrixBase& a, const py::array& b) {
                auto mb = elementwise_operand_matrix_from_numpy(b);
                maybe_warn_float_underpromotion("elementwise_multiply", promotion::BinaryOp::ElementwiseMultiply, a, *mb);
                auto out = pycauset::elementwise_multiply(a, *mb, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const MatrixBase&, const py::object&) -> py::object {
                // Allow other operand types (e.g., internal BlockMatrix) to
                // handle the operation via their __rmul__.
                return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const MatrixBase& a, double s) {
                auto out = a.multiply_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const MatrixBase& a, int64_t s) {
                auto out = a.multiply_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const MatrixBase& a, std::complex<double> s) {
                auto out = a.multiply_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const MatrixBase& a, double s) {
                auto out = a.multiply_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const MatrixBase& a, const py::array& b) {
                auto mb = elementwise_operand_matrix_from_numpy(b);
                maybe_warn_float_underpromotion("elementwise_multiply", promotion::BinaryOp::ElementwiseMultiply, *mb, a);
                auto out = pycauset::elementwise_multiply(*mb, a, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const MatrixBase& a, int64_t s) {
                auto out = a.multiply_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const MatrixBase& a, std::complex<double> s) {
                auto out = a.multiply_scalar(s, "");
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::is_operator());

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

    auto float_matrix = py::class_<DenseMatrix<double>, MatrixBase, std::shared_ptr<DenseMatrix<double>>>(m, "FloatMatrix");
    float_matrix
        .def(
            py::init([](int n) { return std::make_shared<DenseMatrix<double>>(n); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<double>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<double>>(array)) {
                    throw py::type_error("FloatMatrix(np_array): expected dtype float64 and rank-2");
                }
                return dense_matrix_from_numpy_2d<double>(array.cast<py::array_t<double>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<double>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<double>::get)
        .def("set", &DenseMatrix<double>::set)
        .def("__setitem__", [](DenseMatrix<double>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def(
            "inverse",
            [](const DenseMatrix<double>& m) {
                auto out = m.inverse("");
                return std::shared_ptr<DenseMatrix<double>>(out.release());
            })
        .def(
            "invert",
            [](const DenseMatrix<double>& m) {
                auto out = m.inverse("");
                return std::shared_ptr<DenseMatrix<double>>(out.release());
            })
        .def("inverse_to", &DenseMatrix<double>::inverse_to)
        .def("set_identity", &DenseMatrix<double>::set_identity)
        .def("fill", &DenseMatrix<double>::fill)
        .def(
            "matmul",
            [](const DenseMatrix<double>& m,
               const DenseMatrix<double>& other) {
                return std::shared_ptr<DenseMatrix<double>>(m.multiply(other, ""));
            },
            py::arg("other"));

    install_dense_matrix_buffer<double>(float_matrix);

    // --- Float32 Support ---
    auto float32_matrix = py::class_<DenseMatrix<float>, MatrixBase, std::shared_ptr<DenseMatrix<float>>>(m, "Float32Matrix");
    float32_matrix
        .def(
            py::init([](int n) { return std::make_shared<DenseMatrix<float>>(n); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<float>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<float>>(array)) {
                    throw py::type_error("Float32Matrix(np_array): expected dtype float32 and rank-2");
                }
                return dense_matrix_from_numpy_2d<float>(array.cast<py::array_t<float>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<float>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<float>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<float>::get)
        .def("set", &DenseMatrix<float>::set)
        .def("__setitem__", [](DenseMatrix<float>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def(
            "inverse",
            [](const DenseMatrix<float>& m) {
                auto out = m.inverse("");
                return std::shared_ptr<DenseMatrix<float>>(out.release());
            })
        .def(
            "invert",
            [](const DenseMatrix<float>& m) {
                auto out = m.inverse("");
                return std::shared_ptr<DenseMatrix<float>>(out.release());
            })
        .def("inverse_to", &DenseMatrix<float>::inverse_to)
        .def("set_identity", &DenseMatrix<float>::set_identity)
        .def("fill", &DenseMatrix<float>::fill)
        .def(
            "matmul",
            [](const DenseMatrix<float>& m,
               const DenseMatrix<float>& other) {
                return std::shared_ptr<DenseMatrix<float>>(m.multiply(other, ""));
            },
            py::arg("other"));

    install_dense_matrix_buffer<float>(float32_matrix);

    // --- Float16 Support ---
    auto float16_matrix = py::class_<DenseMatrix<float16_t>, MatrixBase, std::shared_ptr<DenseMatrix<float16_t>>>(m, "Float16Matrix");
    float16_matrix
        .def(
            py::init([](int n) { return std::make_shared<DenseMatrix<float16_t>>(n); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<float16_t>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!is_numpy_float16(array)) {
                    throw py::type_error("Float16Matrix(np_array): expected dtype float16 and rank-2");
                }
                auto buf = array.request();
                require_2d(buf);
                const uint64_t rows = static_cast<uint64_t>(buf.shape[0]);
                const uint64_t cols = static_cast<uint64_t>(buf.shape[1]);
                auto result = std::make_shared<DenseMatrix<float16_t>>(rows, cols);

                const uint16_t* src_ptr = static_cast<const uint16_t*>(buf.ptr);
                float16_t* dst_ptr = result->data();

                const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                for (uint64_t i = 0; i < rows; ++i) {
                    for (uint64_t j = 0; j < cols; ++j) {
                        dst_ptr[i * cols + j] = float16_t(static_cast<uint16_t>(src_ptr[i * stride0 + j * stride1]));
                    }
                }
                return result;
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<float16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<float16_t>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def(
            "get",
            [](const DenseMatrix<float16_t>& mat, uint64_t i, uint64_t j) {
                return static_cast<float>(mat.get(i, j));
            })
        .def(
            "set",
            [](DenseMatrix<float16_t>& mat, uint64_t i, uint64_t j, double value) {
                mat.set(i, j, float16_t(value));
            })
        .def("__setitem__", [](DenseMatrix<float16_t>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def("set_identity", &DenseMatrix<float16_t>::set_identity)
        .def(
            "fill",
            [](DenseMatrix<float16_t>& mat, double v) { mat.fill(float16_t(v)); },
            py::arg("value"))
        .def(
            "matmul",
            [](const DenseMatrix<float16_t>& m,
               const DenseMatrix<float16_t>& other) {
                return std::shared_ptr<DenseMatrix<float16_t>>(m.multiply(other, ""));
            },
            py::arg("other"));

    install_dense_matrix_buffer<float16_t>(float16_matrix);

    // --- Complex Float16 Support (two-plane storage) ---
    py::class_<ComplexFloat16Matrix, MatrixBase, std::shared_ptr<ComplexFloat16Matrix>>(
        m, "ComplexFloat16Matrix")
        .def(
            py::init([](int n) { return std::make_shared<ComplexFloat16Matrix>(static_cast<uint64_t>(n)); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<ComplexFloat16Matrix>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                auto buf = array.request();
                require_2d(buf);
                const uint64_t rows = static_cast<uint64_t>(buf.shape[0]);
                const uint64_t cols = static_cast<uint64_t>(buf.shape[1]);

                auto out = std::make_shared<ComplexFloat16Matrix>(rows, cols);
                auto* rdst = out->real_data();
                auto* idst = out->imag_data();

                if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
                    const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(std::complex<float>)));
                    const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(std::complex<float>)));
                    const auto* src = static_cast<const std::complex<float>*>(buf.ptr);
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            const std::complex<float> z = src[i * stride0 + j * stride1];
                            const uint64_t idx = i * cols + j;
                            rdst[idx] = float16_t(static_cast<double>(z.real()));
                            idst[idx] = float16_t(static_cast<double>(z.imag()));
                        }
                    }
                    return out;
                }

                if (py::isinstance<py::array_t<std::complex<double>>>(array)) {
                    const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(std::complex<double>)));
                    const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(std::complex<double>)));
                    const auto* src = static_cast<const std::complex<double>*>(buf.ptr);
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t j = 0; j < cols; ++j) {
                            const std::complex<double> z = src[i * stride0 + j * stride1];
                            const uint64_t idx = i * cols + j;
                            rdst[idx] = float16_t(z.real());
                            idst[idx] = float16_t(z.imag());
                        }
                    }
                    return out;
                }

                throw py::type_error("ComplexFloat16Matrix(np_array): expected dtype complex64 or complex128 and rank-2");
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<ComplexFloat16Matrix>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<ComplexFloat16Matrix>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &ComplexFloat16Matrix::get)
        .def("set", &ComplexFloat16Matrix::set)
        .def(
            "fill",
            [](ComplexFloat16Matrix& mat, std::complex<double> v) { mat.fill(v); },
            py::arg("value"))
        .def("__setitem__", [](ComplexFloat16Matrix& mat, const py::object& key, std::complex<double> value) {
            auto [i, j] = parse_matrix_index(key, mat.rows(), mat.cols());
            mat.set(i, j, value);
        });

    // --- Complex Float32/64 Support ---
    py::class_<DenseMatrix<std::complex<float>>, MatrixBase, std::shared_ptr<DenseMatrix<std::complex<float>>>>(
        m, "ComplexFloat32Matrix")
        .def(
            py::init([](int n) { return std::make_shared<DenseMatrix<std::complex<float>>>(static_cast<uint64_t>(n)); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<std::complex<float>>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<std::complex<float>>>(array)) {
                    throw py::type_error("ComplexFloat32Matrix(np_array): expected dtype complex64 and rank-2");
                }
                return dense_matrix_from_numpy_2d<std::complex<float>>(array.cast<py::array_t<std::complex<float>>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<std::complex<float>>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<std::complex<float>>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<std::complex<float>>::get)
        .def("set", &DenseMatrix<std::complex<float>>::set)
        .def("fill", &DenseMatrix<std::complex<float>>::fill)
        .def("set_identity", &DenseMatrix<std::complex<float>>::set_identity)
        .def("__setitem__", [](DenseMatrix<std::complex<float>>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        });

    py::class_<DenseMatrix<std::complex<double>>, MatrixBase, std::shared_ptr<DenseMatrix<std::complex<double>>>>(
        m, "ComplexFloat64Matrix")
        .def(
            py::init([](int n) { return std::make_shared<DenseMatrix<std::complex<double>>>(static_cast<uint64_t>(n)); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<std::complex<double>>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<std::complex<double>>>(array)) {
                    throw py::type_error("ComplexFloat64Matrix(np_array): expected dtype complex128 and rank-2");
                }
                return dense_matrix_from_numpy_2d<std::complex<double>>(array.cast<py::array_t<std::complex<double>>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<std::complex<double>>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<std::complex<double>>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<std::complex<double>>::get)
        .def("set", &DenseMatrix<std::complex<double>>::set)
        .def("fill", &DenseMatrix<std::complex<double>>::fill)
        .def("set_identity", &DenseMatrix<std::complex<double>>::set_identity)
        .def("__setitem__", [](DenseMatrix<std::complex<double>>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        });

    // --- Diagonal matrix ---
    py::class_<DiagonalMatrix<double>, MatrixBase, std::shared_ptr<DiagonalMatrix<double>>>(m, "DiagonalMatrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DiagonalMatrix<double>>(n, ""); }),
            py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DiagonalMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DiagonalMatrix<double>::get)
        .def("set", &DiagonalMatrix<double>::set)
        .def("get_diagonal", &DiagonalMatrix<double>::get_diagonal)
        .def("set_diagonal", &DiagonalMatrix<double>::set_diagonal)
        .def("__setitem__", [](DiagonalMatrix<double>& mat, const py::object& key, double value) {
            auto [i, j] = parse_matrix_index(key, mat.rows(), mat.cols());
            mat.set(i, j, value);
        });

    // --- Identity matrix ---
    py::class_<IdentityMatrix<double>, MatrixBase, std::shared_ptr<IdentityMatrix<double>>>(m, "IdentityMatrix")
        .def(
            py::init([](py::object x) {
                // Supported forms:
                // - int N -> NxN
                // - sequence [rows, cols] -> rows x cols
                // - MatrixBase -> rows x cols
                // - VectorBase -> N x N
                if (py::isinstance<py::bool_>(x)) {
                    throw py::type_error("IdentityMatrix(x): bool is not a valid size");
                }

                if (py::isinstance<py::int_>(x)) {
                    uint64_t n = x.cast<uint64_t>();
                    return std::make_shared<IdentityMatrix<double>>(n, "");
                }

                // Matrix-like (native)
                try {
                    auto m_in = x.cast<std::shared_ptr<MatrixBase>>();
                    if (m_in) {
                        return std::make_shared<IdentityMatrix<double>>(m_in->rows(), m_in->cols(), "");
                    }
                } catch (...) {
                }

                // Vector-like (native)
                try {
                    auto v_in = x.cast<std::shared_ptr<VectorBase>>();
                    if (v_in) {
                        return std::make_shared<IdentityMatrix<double>>(v_in->size(), "");
                    }
                } catch (...) {
                }

                // Sequence [rows, cols]
                if (py::isinstance<py::sequence>(x)) {
                    py::sequence seq = x.cast<py::sequence>();
                    if (py::len(seq) == 2) {
                        py::object r0 = seq[0];
                        py::object c0 = seq[1];
                        if (!py::isinstance<py::int_>(r0) || !py::isinstance<py::int_>(c0)) {
                            throw py::type_error("IdentityMatrix([rows, cols]) requires two integers");
                        }
                        uint64_t rows = r0.cast<uint64_t>();
                        uint64_t cols = c0.cast<uint64_t>();
                        return std::make_shared<IdentityMatrix<double>>(rows, cols, "");
                    }
                }

                // Duck-typed matrix/vector: rows/cols or size
                if (py::hasattr(x, "rows") && py::hasattr(x, "cols")) {
                    py::object rows_attr = x.attr("rows");
                    py::object cols_attr = x.attr("cols");
                    uint64_t rows = PyCallable_Check(rows_attr.ptr()) ? rows_attr().cast<py::function>()().cast<uint64_t>() : rows_attr.cast<uint64_t>();
                    uint64_t cols = PyCallable_Check(cols_attr.ptr()) ? cols_attr().cast<py::function>()().cast<uint64_t>() : cols_attr.cast<uint64_t>();
                    return std::make_shared<IdentityMatrix<double>>(rows, cols, "");
                }

                if (py::hasattr(x, "size")) {
                    py::object size_attr = x.attr("size");
                    uint64_t n = PyCallable_Check(size_attr.ptr()) ? size_attr().cast<py::function>()().cast<uint64_t>() : size_attr.cast<uint64_t>();
                    return std::make_shared<IdentityMatrix<double>>(n, "");
                }

                throw py::type_error("IdentityMatrix(x) expects an int, [rows, cols], or a matrix/vector");
            }),
            py::arg("x"))
        .def(
            py::init([](uint64_t n) { return std::make_shared<IdentityMatrix<double>>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<IdentityMatrix<double>>(rows, cols, ""); }),
            py::arg("rows"),
            py::arg("cols"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<IdentityMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<IdentityMatrix<double>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &IdentityMatrix<double>::get);

    // --- Integer + Bit matrices (names expected by Python) ---
    py::class_<DenseMatrix<int32_t>, MatrixBase, std::shared_ptr<DenseMatrix<int32_t>>>(m, "IntegerMatrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DenseMatrix<int32_t>>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<int32_t>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<int32_t>>(array)) {
                    throw py::type_error("IntegerMatrix(np_array): expected dtype int32 and rank-2");
                }
                return dense_matrix_from_numpy_2d<int32_t>(array.cast<py::array_t<int32_t>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<int32_t>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<int32_t>::get)
        .def("set", &DenseMatrix<int32_t>::set)
        .def("__getitem__", [](const DenseMatrix<int32_t>& mat, const py::object& key) {
            return dense_getitem(mat, key);
        })
        .def("__setitem__", [](DenseMatrix<int32_t>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def("fill", &DenseMatrix<int32_t>::fill)
        .def(
            "invert",
            [](const DenseMatrix<int32_t>& /*m*/) {
                throw std::runtime_error("Inverse not implemented for IntegerMatrix");
            })
        .def(
            "__invert__",
            [](const DenseMatrix<int32_t>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<DenseMatrix<int32_t>>(out.release());
            },
            py::is_operator());

    py::class_<DenseMatrix<int16_t>, MatrixBase, std::shared_ptr<DenseMatrix<int16_t>>>(m, "Int16Matrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DenseMatrix<int16_t>>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<int16_t>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<int16_t>>(array)) {
                    throw py::type_error("Int16Matrix(np_array): expected dtype int16 and rank-2");
                }
                return dense_matrix_from_numpy_2d<int16_t>(array.cast<py::array_t<int16_t>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<int16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<int16_t>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<int16_t>::get)
        .def("set", &DenseMatrix<int16_t>::set)
        .def("__getitem__", [](const DenseMatrix<int16_t>& mat, const py::object& key) {
            return dense_getitem(mat, key);
        })
        .def("__setitem__", [](DenseMatrix<int16_t>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def("fill", &DenseMatrix<int16_t>::fill)
        .def(
            "__invert__",
            [](const DenseMatrix<int16_t>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<DenseMatrix<int16_t>>(out.release());
            },
            py::is_operator());

    py::class_<DenseMatrix<int8_t>, MatrixBase, std::shared_ptr<DenseMatrix<int8_t>>>(m, "Int8Matrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DenseMatrix<int8_t>>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<int8_t>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<int8_t>>(array)) {
                    throw py::type_error("Int8Matrix(np_array): expected dtype int8 and rank-2");
                }
                return dense_matrix_from_numpy_2d<int8_t>(array.cast<py::array_t<int8_t>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<int8_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<int8_t>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<int8_t>::get)
        .def("set", &DenseMatrix<int8_t>::set)
        .def("__getitem__", [](const DenseMatrix<int8_t>& mat, const py::object& key) {
            return dense_getitem(mat, key);
        })
        .def("fill", &DenseMatrix<int8_t>::fill)
        .def("__setitem__", [](DenseMatrix<int8_t>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def(
            "__invert__",
            [](const DenseMatrix<int8_t>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<DenseMatrix<int8_t>>(out.release());
            },
            py::is_operator());

    py::class_<DenseMatrix<int64_t>, MatrixBase, std::shared_ptr<DenseMatrix<int64_t>>>(m, "Int64Matrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DenseMatrix<int64_t>>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<int64_t>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<int64_t>>(array)) {
                    throw py::type_error("Int64Matrix(np_array): expected dtype int64 and rank-2");
                }
                return dense_matrix_from_numpy_2d<int64_t>(array.cast<py::array_t<int64_t>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<int64_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<int64_t>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<int64_t>::get)
        .def("set", &DenseMatrix<int64_t>::set)
        .def("__getitem__", [](const DenseMatrix<int64_t>& mat, const py::object& key) {
            return dense_getitem(mat, key);
        })
        .def("fill", &DenseMatrix<int64_t>::fill)
        .def("__setitem__", [](DenseMatrix<int64_t>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def(
            "__invert__",
            [](const DenseMatrix<int64_t>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<DenseMatrix<int64_t>>(out.release());
            },
            py::is_operator());

    py::class_<DenseMatrix<uint8_t>, MatrixBase, std::shared_ptr<DenseMatrix<uint8_t>>>(m, "UInt8Matrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DenseMatrix<uint8_t>>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<uint8_t>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<uint8_t>>(array)) {
                    throw py::type_error("UInt8Matrix(np_array): expected dtype uint8 and rank-2");
                }
                return dense_matrix_from_numpy_2d<uint8_t>(array.cast<py::array_t<uint8_t>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<uint8_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<uint8_t>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<uint8_t>::get)
        .def("set", &DenseMatrix<uint8_t>::set)
        .def("__getitem__", [](const DenseMatrix<uint8_t>& mat, const py::object& key) {
            return dense_getitem(mat, key);
        })
        .def("fill", &DenseMatrix<uint8_t>::fill)
        .def("__setitem__", [](DenseMatrix<uint8_t>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def(
            "__invert__",
            [](const DenseMatrix<uint8_t>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<DenseMatrix<uint8_t>>(out.release());
            },
            py::is_operator());

    py::class_<DenseMatrix<uint16_t>, MatrixBase, std::shared_ptr<DenseMatrix<uint16_t>>>(m, "UInt16Matrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DenseMatrix<uint16_t>>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<uint16_t>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<uint16_t>>(array)) {
                    throw py::type_error("UInt16Matrix(np_array): expected dtype uint16 and rank-2");
                }
                return dense_matrix_from_numpy_2d<uint16_t>(array.cast<py::array_t<uint16_t>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<uint16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<uint16_t>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<uint16_t>::get)
        .def("set", &DenseMatrix<uint16_t>::set)
        .def("__getitem__", [](const DenseMatrix<uint16_t>& mat, const py::object& key) {
            return dense_getitem(mat, key);
        })
        .def("fill", &DenseMatrix<uint16_t>::fill)
        .def("__setitem__", [](DenseMatrix<uint16_t>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def(
            "__invert__",
            [](const DenseMatrix<uint16_t>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<DenseMatrix<uint16_t>>(out.release());
            },
            py::is_operator());

    py::class_<DenseMatrix<uint32_t>, MatrixBase, std::shared_ptr<DenseMatrix<uint32_t>>>(m, "UInt32Matrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DenseMatrix<uint32_t>>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<uint32_t>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<uint32_t>>(array)) {
                    throw py::type_error("UInt32Matrix(np_array): expected dtype uint32 and rank-2");
                }
                return dense_matrix_from_numpy_2d<uint32_t>(array.cast<py::array_t<uint32_t>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<uint32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<uint32_t>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<uint32_t>::get)
        .def("set", &DenseMatrix<uint32_t>::set)
        .def("__getitem__", [](const DenseMatrix<uint32_t>& mat, const py::object& key) {
            return dense_getitem(mat, key);
        })
        .def("fill", &DenseMatrix<uint32_t>::fill)
        .def("__setitem__", [](DenseMatrix<uint32_t>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def(
            "__invert__",
            [](const DenseMatrix<uint32_t>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<DenseMatrix<uint32_t>>(out.release());
            },
            py::is_operator());

    py::class_<DenseMatrix<uint64_t>, MatrixBase, std::shared_ptr<DenseMatrix<uint64_t>>>(m, "UInt64Matrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DenseMatrix<uint64_t>>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseMatrix<uint64_t>>(rows, cols); }),
            py::arg("rows"),
            py::arg("cols"))
        .def(
            py::init([](const py::array& array) {
                if (!py::isinstance<py::array_t<uint64_t>>(array)) {
                    throw py::type_error("UInt64Matrix(np_array): expected dtype uint64 and rank-2");
                }
                return dense_matrix_from_numpy_2d<uint64_t>(array.cast<py::array_t<uint64_t>>());
            }),
            py::arg("array"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<uint64_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseMatrix<uint64_t>>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseMatrix<uint64_t>::get)
        .def("set", &DenseMatrix<uint64_t>::set)
        .def("__getitem__", [](const DenseMatrix<uint64_t>& mat, const py::object& key) {
            return dense_getitem(mat, key);
        })
        .def("fill", &DenseMatrix<uint64_t>::fill)
        .def("__setitem__", [](DenseMatrix<uint64_t>& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def(
            "__invert__",
            [](const DenseMatrix<uint64_t>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<DenseMatrix<uint64_t>>(out.release());
            },
            py::is_operator());

    py::class_<DenseBitMatrix, MatrixBase, std::shared_ptr<DenseBitMatrix>>(m, "DenseBitMatrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<DenseBitMatrix>(n, ""); }),
            py::arg("n"))
        .def(
            py::init([](uint64_t rows, uint64_t cols) { return std::make_shared<DenseBitMatrix>(rows, cols, ""); }),
            py::arg("rows"),
            py::arg("cols"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseBitMatrix>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "_from_storage",
            [](uint64_t rows,
               uint64_t cols,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseBitMatrix>(rows, cols, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseBitMatrix::get)
        .def("set", [](DenseBitMatrix& mat, uint64_t i, uint64_t j, const py::object& value) {
            mat.set(i, j, coerce_bit_value(value));
        })
        .def(
            "fill",
            [](DenseBitMatrix& mat, const py::object& value) { mat.fill(coerce_bit_value(value)); },
            py::arg("value"))
        .def("__getitem__", [](const DenseBitMatrix& mat, const py::object& key) {
            return dense_getitem(mat, key);
        })
        .def("__setitem__", [](DenseBitMatrix& mat, const py::object& key, const py::object& value) {
            dense_setitem(mat, key, value);
        })
        .def(
            "__invert__",
            [](const DenseBitMatrix& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<DenseBitMatrix>(out.release());
            },
            py::is_operator())
        .def_static(
            "random",
            [](uint64_t n, double p, std::optional<uint64_t> seed) {
                auto out = DenseBitMatrix::random(n, p, "", seed);
                return std::shared_ptr<DenseBitMatrix>(out.release());
            },
            py::arg("n"),
            py::arg("p"),
            py::arg("seed") = std::optional<uint64_t>{})
        .def_static(
            "random",
            [](uint64_t rows, uint64_t cols, double p, std::optional<uint64_t> seed) {
                auto out = DenseBitMatrix::random(rows, cols, p, "", seed);
                return std::shared_ptr<DenseBitMatrix>(out.release());
            },
            py::arg("rows"),
            py::arg("cols"),
            py::arg("p"),
            py::arg("seed") = std::optional<uint64_t>{});

    // --- Triangular matrices ---
    py::class_<TriangularBitMatrix, MatrixBase, std::shared_ptr<TriangularBitMatrix>>(m, "TriangularBitMatrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<TriangularBitMatrix>(n, ""); }),
            py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<TriangularBitMatrix>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &TriangularBitMatrix::get)
        .def("set", [](TriangularBitMatrix& mat, uint64_t i, uint64_t j, const py::object& value) {
            mat.set(i, j, coerce_bit_value(value));
        })
        .def("__getitem__", [](const TriangularBitMatrix& mat, const py::object& key) {
            auto [i, j] = parse_matrix_index(key, mat.rows(), mat.cols());
            return mat.get(i, j) ? 1.0 : 0.0;
        })
        .def("__setitem__", [](TriangularBitMatrix& mat, const py::object& key, const py::object& value) {
            // Support scalar assignment and basic batch assignment (rows, cols) = vals
            if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
                py::sequence seq = key.cast<py::sequence>();
                if (seq.size() == 2 && py::isinstance<py::array>(seq[0]) && py::isinstance<py::array>(seq[1])) {
                    auto rows_arr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>(seq[0]);
                    auto cols_arr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>(seq[1]);
                    auto rows_buf = rows_arr.request();
                    auto cols_buf = cols_arr.request();
                    require_1d(rows_buf);
                    require_1d(cols_buf);
                    if (rows_buf.shape[0] != cols_buf.shape[0]) {
                        throw py::value_error("rows and cols must have the same length");
                    }

                    const auto* rows_ptr = static_cast<const int64_t*>(rows_buf.ptr);
                    const auto* cols_ptr = static_cast<const int64_t*>(cols_buf.ptr);
                    uint64_t count = static_cast<uint64_t>(rows_buf.shape[0]);

                    if (py::isinstance<py::array>(value)) {
                        auto vals_arr = py::array_t<bool, py::array::c_style | py::array::forcecast>(value);
                        auto vals_buf = vals_arr.request();
                        require_1d(vals_buf);
                        if (vals_buf.shape[0] != static_cast<py::ssize_t>(count)) {
                            throw py::value_error("vals must have the same length as rows/cols");
                        }
                        const auto* vals_ptr = static_cast<const bool*>(vals_buf.ptr);
                        for (uint64_t k = 0; k < count; ++k) {
                            uint64_t i = static_cast<uint64_t>(rows_ptr[k]);
                            uint64_t j = static_cast<uint64_t>(cols_ptr[k]);
                            if (i == j) {
                                bindings_warn::warn("Diagonal assignment ignored for TriangularBitMatrix");
                                continue;
                            }
                            mat.set(i, j, vals_ptr[k]);
                        }
                        return;
                    }

                    bool bit = coerce_bit_value(value);
                    for (uint64_t k = 0; k < count; ++k) {
                        uint64_t i = static_cast<uint64_t>(rows_ptr[k]);
                        uint64_t j = static_cast<uint64_t>(cols_ptr[k]);
                        if (i == j) {
                            bindings_warn::warn("Diagonal assignment ignored for TriangularBitMatrix");
                            continue;
                        }
                        mat.set(i, j, bit);
                    }
                    return;
                }
            }

            auto [i, j] = parse_matrix_index(key, mat.rows(), mat.cols());
            if (i == j) {
                bindings_warn::warn("Diagonal assignment ignored for TriangularBitMatrix");
                return;
            }
            mat.set(i, j, coerce_bit_value(value));
        })
        .def(
            "invert",
            [](const TriangularBitMatrix& mat) {
                auto out = mat.inverse("");
                return std::shared_ptr<TriangularMatrix<double>>(out.release());
            })
        .def(
            "__invert__",
            [](const TriangularBitMatrix& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<TriangularBitMatrix>(out.release());
            },
            py::is_operator())
        .def(
            "elementwise_multiply",
            [](const TriangularBitMatrix& a, const TriangularBitMatrix& b) {
                auto out = a.elementwise_multiply(b, "");
                return std::shared_ptr<TriangularBitMatrix>(out.release());
            },
            py::arg("other"))
        .def_static(
            "random",
            [](uint64_t n, double p, std::optional<uint64_t> seed) {
                auto out = TriangularBitMatrix::random(n, p, "", seed);
                return std::shared_ptr<TriangularBitMatrix>(out.release());
            },
            py::arg("n"),
            py::arg("p"),
            py::arg("seed") = std::optional<uint64_t>{});

    py::class_<TriangularMatrix<double>, MatrixBase, std::shared_ptr<TriangularMatrix<double>>>(m, "TriangularFloatMatrix")
        .def(
            py::init([](uint64_t n, bool has_diagonal) {
                return std::make_shared<TriangularMatrix<double>>(n, "", has_diagonal);
            }),
            py::arg("n"),
            py::arg("has_diagonal") = false)
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<TriangularMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &TriangularMatrix<double>::get)
        .def("set", &TriangularMatrix<double>::set)
        .def(
            "invert",
            [](const TriangularMatrix<double>& m) {
                auto out = m.inverse("");
                return std::shared_ptr<TriangularMatrix<double>>(out.release());
            })
        .def("__setitem__", [](TriangularMatrix<double>& mat, const py::object& key, double value) {
            auto [i, j] = parse_matrix_index(key, mat.rows(), mat.cols());
            mat.set(i, j, value);
        })
        .def(
            "__invert__",
            [](const TriangularMatrix<double>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<TriangularMatrix<double>>(out.release());
            },
            py::is_operator());

    py::class_<TriangularMatrix<int32_t>, MatrixBase, std::shared_ptr<TriangularMatrix<int32_t>>>(m, "TriangularIntegerMatrix")
        .def(
            py::init([](uint64_t n, bool has_diagonal) {
                return std::make_shared<TriangularMatrix<int32_t>>(n, "", has_diagonal);
            }),
            py::arg("n"),
            py::arg("has_diagonal") = false)
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<TriangularMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &TriangularMatrix<int32_t>::get)
        .def("set", &TriangularMatrix<int32_t>::set)
        .def("__setitem__", [](TriangularMatrix<int32_t>& mat, const py::object& key, int32_t value) {
            auto [i, j] = parse_matrix_index(key, mat.rows(), mat.cols());
            mat.set(i, j, value);
        })
        .def(
            "__invert__",
            [](const TriangularMatrix<int32_t>& mat) {
                auto out = mat.bitwise_not("");
                return std::shared_ptr<TriangularMatrix<int32_t>>(out.release());
            },
            py::is_operator());

    using SymmetricFloat64Matrix = pycauset::SymmetricMatrix<double>;
    using AntiSymmetricFloat64Matrix = pycauset::AntiSymmetricMatrix<double>;

    py::class_<SymmetricFloat64Matrix, MatrixBase, std::shared_ptr<SymmetricFloat64Matrix>>(m, "SymmetricMatrix")
        .def(
            py::init([](uint64_t n) {
                return std::make_shared<SymmetricFloat64Matrix>(n, "", false);
            }),
            py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<SymmetricFloat64Matrix>(n, backing_file, offset, seed, scalar, is_transposed, false);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("is_antisymmetric", &SymmetricFloat64Matrix::is_antisymmetric)
        .def("get", [](const SymmetricFloat64Matrix& mat, uint64_t i, uint64_t j) { return mat.get(i, j); })
        .def("set", [](SymmetricFloat64Matrix& mat, uint64_t i, uint64_t j, double value) {
            translate_invalid_argument([&]() {
                mat.set(i, j, value);
            });
        })
        .def_static(
            "from_triangular",
            [](const TriangularMatrix<double>& source) {
                auto out = SymmetricFloat64Matrix::from_triangular(source, "");
                return std::shared_ptr<SymmetricFloat64Matrix>(out.release());
            },
            py::arg("source"));

    py::class_<AntiSymmetricFloat64Matrix, SymmetricFloat64Matrix, std::shared_ptr<AntiSymmetricFloat64Matrix>>(m, "AntiSymmetricMatrix")
        .def(
            py::init([](uint64_t n) { return std::make_shared<AntiSymmetricFloat64Matrix>(n, ""); }),
            py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<AntiSymmetricFloat64Matrix>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_static(
            "from_triangular",
            [](const TriangularMatrix<double>& source) {
                auto out = AntiSymmetricFloat64Matrix::from_triangular(source, "");
                return std::shared_ptr<AntiSymmetricFloat64Matrix>(out.release());
            },
            py::arg("source"));

    m.def("asarray", [](const py::array& array) -> py::object {
        auto buf = array.request();
        if (buf.ndim == 1) {
            if (is_numpy_float16(array)) {
                auto b = array.request();
                require_1d(b);
                uint64_t n = static_cast<uint64_t>(b.shape[0]);
                auto result = std::make_shared<DenseVector<float16_t>>(n);
                const uint16_t* src_ptr = static_cast<const uint16_t*>(b.ptr);
                float16_t* dst_ptr = result->data();
                if (b.strides[0] == static_cast<py::ssize_t>(sizeof(uint16_t))) {
                    std::memcpy(dst_ptr, src_ptr, n * sizeof(uint16_t));
                } else {
                    const ptrdiff_t stride0 = static_cast<ptrdiff_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                    for (uint64_t i = 0; i < n; ++i) {
                        const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0;
                        dst_ptr[i] = float16_t(static_cast<uint16_t>(src_ptr[idx]));
                    }
                }
                return py::cast(result);
            }
            if (py::isinstance<py::array_t<double>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<double>(array.cast<py::array_t<double>>()));
            }
            if (py::isinstance<py::array_t<float>>(array)) {
                // Promote float32 vectors to float64 vectors
                return py::cast(dense_vector_from_numpy_1d_float32_promote_to_float64(array.cast<py::array_t<float>>()));
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
                maybe_warn_integer_matmul_accumulator_widen(*a, *b);
                maybe_warn_integer_matmul_overflow_risk_preflight(*a, *b);
                auto out = pycauset::dispatch_matmul(*a, *b, "");
                return std::shared_ptr<MatrixBase>(out.release());
            });
        },
        py::arg("a"),
        py::arg("b"));

    m.def(
        "compute_k_matrix",
        [](std::shared_ptr<TriangularBitMatrix> C,
           double a,
           int num_threads) {
            if (!C) {
                throw std::invalid_argument("C must not be None");
            }
            auto out = ComputeContext::instance().get_device()->compute_k_matrix(*C, a, "", num_threads);
            return std::shared_ptr<TriangularMatrix<double>>(out.release());
        },
        py::arg("C"),
        py::arg("a"),
        py::arg("num_threads") = 0);

}
