#include "bindings_common.hpp"

#include "binding_warnings.hpp"

#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/core/Float16.hpp"
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

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

using namespace pycauset;

namespace {

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

    if (buf.strides[1] == static_cast<py::ssize_t>(sizeof(T)) &&
        buf.strides[0] == static_cast<py::ssize_t>(cols * sizeof(T))) {
        std::memcpy(dst_ptr, src_ptr, rows * cols * sizeof(T));
    } else {
        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
        const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(T)));
        for (uint64_t i = 0; i < rows; ++i) {
            for (uint64_t j = 0; j < cols; ++j) {
                dst_ptr[i * cols + j] = src_ptr[i * stride0 + j * stride1];
            }
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
    const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
    if (stride0 == 1) {
        std::memcpy(dst_ptr, src_ptr, n * sizeof(T));
    } else {
        for (uint64_t i = 0; i < n; ++i) {
            dst_ptr[i] = src_ptr[i * stride0];
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
    const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
    if (stride0 == 1) {
        std::memcpy(dst_ptr, src_ptr, cols * sizeof(T));
    } else {
        for (uint64_t j = 0; j < cols; ++j) {
            dst_ptr[j] = src_ptr[j * stride0];
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
        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, src_ptr[i * stride0]);
        }
        return result;
    }
    if (py::isinstance<py::array_t<float>>(array)) {
        // Promote float32 vectors to float64 vectors for now.
        auto tmp = array.cast<py::array_t<float>>();
        auto buf = tmp.request();
        require_1d(buf);
        uint64_t n = static_cast<uint64_t>(buf.shape[0]);
        auto result = std::make_shared<DenseVector<double>>(n);
        const float* src_ptr = static_cast<const float*>(buf.ptr);
        double* dst_ptr = result->data();
        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(float)));
        for (uint64_t i = 0; i < n; ++i) {
            dst_ptr[i] = static_cast<double>(src_ptr[i * stride0]);
        }
        return result;
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

        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
        const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(uint16_t)));
        for (uint64_t i = 0; i < rows; ++i) {
            for (uint64_t j = 0; j < cols; ++j) {
                dst_ptr[i * cols + j] = float16_t(static_cast<uint16_t>(src_ptr[i * stride0 + j * stride1]));
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
        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
        const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(bool)));
        for (uint64_t i = 0; i < rows; ++i) {
            for (uint64_t j = 0; j < cols; ++j) {
                result->set(i, j, src_ptr[i * stride0 + j * stride1]);
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

        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
        for (uint64_t j = 0; j < cols; ++j) {
            dst_ptr[j] = float16_t(static_cast<uint16_t>(src_ptr[j * stride0]));
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
        const auto stride0 = static_cast<uint64_t>(tmp_buf.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
        for (uint64_t j = 0; j < cols; ++j) {
            result->set(0, j, src_ptr[j * stride0]);
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

inline std::pair<uint64_t, uint64_t> parse_matrix_index(const py::handle& key) {
    if (!py::isinstance<py::tuple>(key) && !py::isinstance<py::list>(key)) {
        throw py::type_error("Matrix indices must be a tuple (i, j)");
    }
    py::sequence seq = key.cast<py::sequence>();
    if (seq.size() != 2) {
        throw py::type_error("Matrix indices must be a tuple (i, j)");
    }
    uint64_t i = seq[0].cast<uint64_t>();
    uint64_t j = seq[1].cast<uint64_t>();
    return {i, j};
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
            if (py::hasattr(self, "cached_trace")) {
                py::object v = self.attr("cached_trace");
                if (!v.is_none()) {
                    return v.cast<double>();
                }
            }
            if (auto cached = mat.get_cached_trace()) {
                double v = *cached;
                self.attr("cached_trace") = v;
                return v;
            }
            double v = pycauset::trace(mat);
            mat.set_cached_trace(v);
            self.attr("cached_trace") = v;
            return v;
        })
        .def("determinant", [](py::object self) {
            auto& mat = self.cast<MatrixBase&>();
            if (py::hasattr(self, "cached_determinant")) {
                py::object v = self.attr("cached_determinant");
                if (!v.is_none()) {
                    return v.cast<double>();
                }
            }
            if (auto cached = mat.get_cached_determinant()) {
                double v = *cached;
                self.attr("cached_determinant") = v;
                return v;
            }
            double v = pycauset::determinant(mat);
            mat.set_cached_determinant(v);
            self.attr("cached_determinant") = v;
            return v;
        })
        .def("__getitem__", [](const MatrixBase& mat, const py::object& key) -> py::object {
            auto [i, j] = parse_matrix_index(key);
            const DataType dt = mat.get_data_type();
            if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
                return py::cast(mat.get_element_as_complex(i, j));
            }
            return py::float_(mat.get_element_as_double(i, j));
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

    py::class_<DenseMatrix<double>, MatrixBase, std::shared_ptr<DenseMatrix<double>>>(
        m, "FloatMatrix", py::buffer_protocol())
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
        .def("__setitem__", [](DenseMatrix<double>& mat, const py::object& key, double value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
        })
        .def_buffer([](DenseMatrix<double>& m) -> py::buffer_info {
            py::ssize_t stride0 = static_cast<py::ssize_t>(sizeof(double) * m.base_cols());
            py::ssize_t stride1 = static_cast<py::ssize_t>(sizeof(double));
            if (m.is_transposed()) {
                std::swap(stride0, stride1);
            }
            return py::buffer_info(
                m.data(),                                 /* Pointer to buffer */
                sizeof(double),                           /* Size of one scalar */
                py::format_descriptor<double>::format(),  /* Python struct-style format descriptor */
                2,                                        /* Number of dimensions */
                {m.rows(), m.cols()},                     /* Buffer dimensions */
                {stride0, stride1});
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

    // --- Float32 Support ---
    py::class_<DenseMatrix<float>, MatrixBase, std::shared_ptr<DenseMatrix<float>>>(
        m, "Float32Matrix", py::buffer_protocol())
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
        .def("__setitem__", [](DenseMatrix<float>& mat, const py::object& key, float value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
        })
        .def_buffer([](DenseMatrix<float>& m) -> py::buffer_info {
            py::ssize_t stride0 = static_cast<py::ssize_t>(sizeof(float) * m.base_cols());
            py::ssize_t stride1 = static_cast<py::ssize_t>(sizeof(float));
            if (m.is_transposed()) {
                std::swap(stride0, stride1);
            }
            return py::buffer_info(
                m.data(),                                /* Pointer to buffer */
                sizeof(float),                           /* Size of one scalar */
                py::format_descriptor<float>::format(),  /* Python struct-style format descriptor */
                2,                                       /* Number of dimensions */
                {m.rows(), m.cols()},                    /* Buffer dimensions */
                {stride0, stride1});
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

    // --- Float16 Support ---
    py::class_<DenseMatrix<float16_t>, MatrixBase, std::shared_ptr<DenseMatrix<float16_t>>>(
        m, "Float16Matrix", py::buffer_protocol())
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
        .def("__setitem__", [](DenseMatrix<float16_t>& mat, const py::object& key, double value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, float16_t(value));
        })
        .def_buffer([](DenseMatrix<float16_t>& m) -> py::buffer_info {
            py::ssize_t stride0 = static_cast<py::ssize_t>(sizeof(float16_t) * m.base_cols());
            py::ssize_t stride1 = static_cast<py::ssize_t>(sizeof(float16_t));
            if (m.is_transposed()) {
                std::swap(stride0, stride1);
            }
            return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                sizeof(float16_t),                      /* Size of one scalar */
                "e",                                    /* PEP 3118: float16 */
                2,                                      /* Number of dimensions */
                {m.rows(), m.cols()},                   /* Buffer dimensions */
                {stride0, stride1});
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
            auto [i, j] = parse_matrix_index(key);
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
        .def("__setitem__", [](DenseMatrix<std::complex<float>>& mat, const py::object& key, std::complex<float> value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
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
        .def("__setitem__", [](DenseMatrix<std::complex<double>>& mat, const py::object& key, std::complex<double> value) {
            auto [i, j] = parse_matrix_index(key);
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
        .def(
            "invert",
            [](const DenseMatrix<int32_t>& /*m*/) {
                throw std::runtime_error("Inverse not implemented for IntegerMatrix");
            })
        .def("__getitem__", [](const DenseMatrix<int32_t>& mat, const py::object& key) {
            auto [i, j] = parse_matrix_index(key);
            return mat.get(i, j);
        })
        .def("fill", &DenseMatrix<int32_t>::fill)
        .def("__setitem__", [](DenseMatrix<int32_t>& mat, const py::object& key, int32_t value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
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
            auto [i, j] = parse_matrix_index(key);
            return mat.get(i, j);
        })
        .def("fill", &DenseMatrix<int16_t>::fill)
        .def("__setitem__", [](DenseMatrix<int16_t>& mat, const py::object& key, int16_t value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
        })
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
            auto [i, j] = parse_matrix_index(key);
            return mat.get(i, j);
        })
        .def("fill", &DenseMatrix<int8_t>::fill)
        .def("__setitem__", [](DenseMatrix<int8_t>& mat, const py::object& key, int8_t value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
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
            auto [i, j] = parse_matrix_index(key);
            return mat.get(i, j);
        })
        .def("fill", &DenseMatrix<int64_t>::fill)
        .def("__setitem__", [](DenseMatrix<int64_t>& mat, const py::object& key, int64_t value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
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
            auto [i, j] = parse_matrix_index(key);
            return mat.get(i, j);
        })
        .def("fill", &DenseMatrix<uint8_t>::fill)
        .def("__setitem__", [](DenseMatrix<uint8_t>& mat, const py::object& key, uint8_t value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
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
            auto [i, j] = parse_matrix_index(key);
            return mat.get(i, j);
        })
        .def("fill", &DenseMatrix<uint16_t>::fill)
        .def("__setitem__", [](DenseMatrix<uint16_t>& mat, const py::object& key, uint16_t value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
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
            auto [i, j] = parse_matrix_index(key);
            return mat.get(i, j);
        })
        .def("fill", &DenseMatrix<uint32_t>::fill)
        .def("__setitem__", [](DenseMatrix<uint32_t>& mat, const py::object& key, uint32_t value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
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
            auto [i, j] = parse_matrix_index(key);
            return mat.get(i, j);
        })
        .def("fill", &DenseMatrix<uint64_t>::fill)
        .def("__setitem__", [](DenseMatrix<uint64_t>& mat, const py::object& key, uint64_t value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, value);
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
            auto [i, j] = parse_matrix_index(key);
            return mat.get(i, j);
        })
        .def("__setitem__", [](DenseBitMatrix& mat, const py::object& key, const py::object& value) {
            auto [i, j] = parse_matrix_index(key);
            mat.set(i, j, coerce_bit_value(value));
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
            auto [i, j] = parse_matrix_index(key);
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

            auto [i, j] = parse_matrix_index(key);
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
            auto [i, j] = parse_matrix_index(key);
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
            auto [i, j] = parse_matrix_index(key);
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
                const auto stride0 = static_cast<uint64_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                for (uint64_t i = 0; i < n; ++i) {
                    dst_ptr[i] = float16_t(static_cast<uint16_t>(src_ptr[i * stride0]));
                }
                return py::cast(result);
            }
            if (py::isinstance<py::array_t<double>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<double>(array.cast<py::array_t<double>>()));
            }
            if (py::isinstance<py::array_t<float>>(array)) {
                // Promote float32 vectors to float64 vectors
                auto tmp = array.cast<py::array_t<float>>();
                auto b = tmp.request();
                require_1d(b);
                uint64_t n = static_cast<uint64_t>(b.shape[0]);
                auto result = std::make_shared<DenseVector<double>>(n);
                const float* src_ptr = static_cast<const float*>(b.ptr);
                double* dst_ptr = result->data();
                const auto stride0 = static_cast<uint64_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(float)));
                for (uint64_t i = 0; i < n; ++i) {
                    dst_ptr[i] = static_cast<double>(src_ptr[i * stride0]);
                }
                return py::cast(result);
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
                const auto stride0 = static_cast<uint64_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
                for (uint64_t i = 0; i < n; ++i) {
                    result->set(i, src_ptr[i * stride0]);
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
                const auto stride0 = static_cast<uint64_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
                const auto stride1 = static_cast<uint64_t>(b.strides[1] / static_cast<py::ssize_t>(sizeof(bool)));
                for (uint64_t i = 0; i < rows; ++i) {
                    for (uint64_t j = 0; j < cols; ++j) {
                        result->set(i, j, src_ptr[i * stride0 + j * stride1]);
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
