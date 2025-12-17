#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/MatrixTypeResolver.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/compute/ComputeDevice.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <functional>
#include <iostream>
#include <complex>

namespace pycauset {

using IntegerMatrix = DenseMatrix<int32_t>;
using FloatMatrix = DenseMatrix<double>;
using Int16Matrix = DenseMatrix<int16_t>;
using Float16Matrix = DenseMatrix<float16_t>;

// --- Helpers ---

namespace {
    inline uint64_t broadcast_dim(uint64_t a, uint64_t b, const char* opname, uint64_t a_other, uint64_t b_other, bool rows) {
        if (a == b) return a;
        if (a == 1) return b;
        if (b == 1) return a;
        std::string msg = std::string(opname) + ": operands could not be broadcast together with shapes (" +
                          std::to_string(a_other) + "," + std::to_string(rows ? a : a_other) + ") and (" +
                          std::to_string(b_other) + "," + std::to_string(rows ? b : b_other) + ")";
        throw std::invalid_argument(msg);
    }

    inline std::pair<uint64_t, uint64_t> broadcast_shape_2d(const MatrixBase& a, const MatrixBase& b, const char* opname) {
        const uint64_t ar = a.rows();
        const uint64_t ac = a.cols();
        const uint64_t br = b.rows();
        const uint64_t bc = b.cols();
        if (ar == br && ac == bc) return {ar, ac};
        // Numpy-style 2D broadcasting.
        const uint64_t rr = (ar == br) ? ar : (ar == 1 ? br : (br == 1 ? ar : 0));
        const uint64_t rc = (ac == bc) ? ac : (ac == 1 ? bc : (bc == 1 ? ac : 0));
        if (rr == 0 || rc == 0) {
            std::string msg = std::string(opname) + ": operands could not be broadcast together with shapes (" +
                              std::to_string(ar) + "," + std::to_string(ac) + ") and (" +
                              std::to_string(br) + "," + std::to_string(bc) + ")";
            throw std::invalid_argument(msg);
        }
        return {rr, rc};
    }
}

bool is_triangular(const MatrixBase& m) {
    return dynamic_cast<const TriangularMatrixBase*>(&m) != nullptr;
}

bool is_identity(const MatrixBase& m) {
    return m.get_matrix_type() == MatrixType::IDENTITY;
}

// --- Matrix Operations ---

std::unique_ptr<MatrixBase> add(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (is_identity(a) && is_identity(b)) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw std::invalid_argument("add: identity operands must have identical shapes");
        }
        if (a.get_data_type() == DataType::FLOAT64 && b.get_data_type() == DataType::FLOAT64) {
            const auto& a_id = static_cast<const IdentityMatrix<double>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<double>&>(b);
            return a_id.add(b_id, result_file);
        }
        if (a.get_data_type() == DataType::INT32 && b.get_data_type() == DataType::INT32) {
            const auto& a_id = static_cast<const IdentityMatrix<int32_t>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<int32_t>&>(b);
            return a_id.add(b_id, result_file);
        }
    }

    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "add");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    MatrixType mtype_a = a.get_matrix_type();
    MatrixType mtype_b = b.get_matrix_type();

    DataType res_dtype = promotion::resolve(promotion::BinaryOp::Add, type_a, type_b).result_dtype;
    MatrixType res_mtype = matrix_promotion::resolve(mtype_a, mtype_b);

    auto result = ObjectFactory::create_matrix(out_rows, out_cols, res_dtype, res_mtype, result_file);
    
    ComputeContext::instance().get_device()->add(a, b, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> subtract(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (is_identity(a) && is_identity(b)) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw std::invalid_argument("subtract: identity operands must have identical shapes");
        }
        if (a.get_data_type() == DataType::FLOAT64 && b.get_data_type() == DataType::FLOAT64) {
            const auto& a_id = static_cast<const IdentityMatrix<double>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<double>&>(b);
            return a_id.subtract(b_id, result_file);
        }
        if (a.get_data_type() == DataType::INT32 && b.get_data_type() == DataType::INT32) {
            const auto& a_id = static_cast<const IdentityMatrix<int32_t>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<int32_t>&>(b);
            return a_id.subtract(b_id, result_file);
        }
    }

    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "subtract");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    MatrixType mtype_a = a.get_matrix_type();
    MatrixType mtype_b = b.get_matrix_type();

    DataType res_dtype = promotion::resolve(promotion::BinaryOp::Subtract, type_a, type_b).result_dtype;
    MatrixType res_mtype = matrix_promotion::resolve(mtype_a, mtype_b);

    auto result = ObjectFactory::create_matrix(out_rows, out_cols, res_dtype, res_mtype, result_file);
    
    ComputeContext::instance().get_device()->subtract(a, b, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> elementwise_multiply(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (is_identity(a) && is_identity(b)) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw std::invalid_argument("elementwise_multiply: identity operands must have identical shapes");
        }
        if (a.get_data_type() == DataType::FLOAT64 && b.get_data_type() == DataType::FLOAT64) {
            const auto& a_id = static_cast<const IdentityMatrix<double>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<double>&>(b);
            return a_id.elementwise_multiply(b_id, result_file);
        }
        if (a.get_data_type() == DataType::INT32 && b.get_data_type() == DataType::INT32) {
            const auto& a_id = static_cast<const IdentityMatrix<int32_t>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<int32_t>&>(b);
            return a_id.elementwise_multiply(b_id, result_file);
        }
    }

    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "elementwise_multiply");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    MatrixType mtype_a = a.get_matrix_type();
    MatrixType mtype_b = b.get_matrix_type();

    DataType res_dtype = promotion::resolve(promotion::BinaryOp::ElementwiseMultiply, type_a, type_b).result_dtype;
    MatrixType res_mtype = matrix_promotion::resolve(mtype_a, mtype_b);

    auto result = ObjectFactory::create_matrix(out_rows, out_cols, res_dtype, res_mtype, result_file);
    
    ComputeContext::instance().get_device()->elementwise_multiply(a, b, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> elementwise_divide(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "elementwise_divide");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    DataType res_dtype = promotion::resolve(promotion::BinaryOp::Divide, type_a, type_b).result_dtype;
    // Division is not generally structure-preserving (e.g., implicit 0/0 off-triangle).
    // Materialize a dense result for correctness.
    MatrixType res_mtype = MatrixType::DENSE_FLOAT;

    auto result = ObjectFactory::create_matrix(out_rows, out_cols, res_dtype, res_mtype, result_file);

    ComputeContext::instance().get_device()->elementwise_divide(a, b, *result);

    return result;
}

// --- Vector Operations ---

std::unique_ptr<VectorBase> add_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();

    const DataType res_dtype = promotion::resolve(
        promotion::BinaryOp::Add,
        a.get_data_type(),
        b.get_data_type()).result_dtype;

    std::unique_ptr<VectorBase> result;
    if (res_dtype == DataType::BIT) {
        result = std::make_unique<DenseVector<bool>>(n, result_file);
    } else if (res_dtype == DataType::INT8) {
        result = std::make_unique<DenseVector<int8_t>>(n, result_file);
    } else if (res_dtype == DataType::INT16) {
        result = std::make_unique<DenseVector<int16_t>>(n, result_file);
    } else if (res_dtype == DataType::INT32) {
        result = std::make_unique<DenseVector<int32_t>>(n, result_file);
    } else if (res_dtype == DataType::INT64) {
        result = std::make_unique<DenseVector<int64_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT8) {
        result = std::make_unique<DenseVector<uint8_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT16) {
        result = std::make_unique<DenseVector<uint16_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT32) {
        result = std::make_unique<DenseVector<uint32_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT64) {
        result = std::make_unique<DenseVector<uint64_t>>(n, result_file);
    } else if (res_dtype == DataType::FLOAT16) {
        result = std::make_unique<DenseVector<float16_t>>(n, result_file);
    } else if (res_dtype == DataType::FLOAT32) {
        result = std::make_unique<DenseVector<float>>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT16) {
        result = std::make_unique<ComplexFloat16Vector>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT32) {
        result = std::make_unique<DenseVector<std::complex<float>>>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT64) {
        result = std::make_unique<DenseVector<std::complex<double>>>(n, result_file);
    } else {
        result = std::make_unique<DenseVector<double>>(n, result_file);
    }

    ComputeContext::instance().get_device()->add_vector(a, b, *result);
    return result;
}

std::unique_ptr<VectorBase> subtract_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();

    const DataType res_dtype = promotion::resolve(
        promotion::BinaryOp::Subtract,
        a.get_data_type(),
        b.get_data_type()).result_dtype;

    std::unique_ptr<VectorBase> result;
    if (res_dtype == DataType::BIT) {
        result = std::make_unique<DenseVector<bool>>(n, result_file);
    } else if (res_dtype == DataType::INT8) {
        result = std::make_unique<DenseVector<int8_t>>(n, result_file);
    } else if (res_dtype == DataType::INT16) {
        result = std::make_unique<DenseVector<int16_t>>(n, result_file);
    } else if (res_dtype == DataType::INT32) {
        result = std::make_unique<DenseVector<int32_t>>(n, result_file);
    } else if (res_dtype == DataType::INT64) {
        result = std::make_unique<DenseVector<int64_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT8) {
        result = std::make_unique<DenseVector<uint8_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT16) {
        result = std::make_unique<DenseVector<uint16_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT32) {
        result = std::make_unique<DenseVector<uint32_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT64) {
        result = std::make_unique<DenseVector<uint64_t>>(n, result_file);
    } else if (res_dtype == DataType::FLOAT16) {
        result = std::make_unique<DenseVector<float16_t>>(n, result_file);
    } else if (res_dtype == DataType::FLOAT32) {
        result = std::make_unique<DenseVector<float>>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT16) {
        result = std::make_unique<ComplexFloat16Vector>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT32) {
        result = std::make_unique<DenseVector<std::complex<float>>>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT64) {
        result = std::make_unique<DenseVector<std::complex<double>>>(n, result_file);
    } else {
        result = std::make_unique<DenseVector<double>>(n, result_file);
    }

    ComputeContext::instance().get_device()->subtract_vector(a, b, *result);
    return result;
}

double dot_product(const VectorBase& a, const VectorBase& b) {
    const auto dt_a = a.get_data_type();
    const auto dt_b = b.get_data_type();
    if (dt_a == DataType::COMPLEX_FLOAT16 || dt_a == DataType::COMPLEX_FLOAT32 || dt_a == DataType::COMPLEX_FLOAT64 ||
        dt_b == DataType::COMPLEX_FLOAT16 || dt_b == DataType::COMPLEX_FLOAT32 || dt_b == DataType::COMPLEX_FLOAT64) {
        throw std::invalid_argument("dot_product: complex vectors not supported yet");
    }
    return ComputeContext::instance().get_device()->dot(a, b);
}

double norm(const VectorBase& v) {
    return ComputeContext::instance().get_device()->l2_norm(v);
}

double norm(const MatrixBase& m) {
    return ComputeContext::instance().get_device()->frobenius_norm(m);
}

std::complex<double> dot_product_complex(const VectorBase& a, const VectorBase& b) {
    const uint64_t n = a.size();
    if (b.size() != n) {
        throw std::invalid_argument("Vector dimensions mismatch");
    }

    // Semantics: simple sum_i a[i] * b[i] (no conjugation).
    double re = 0.0;
    double im = 0.0;

    #pragma omp parallel for reduction(+:re, im)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        const std::complex<double> prod = a.get_element_as_complex(static_cast<uint64_t>(i)) *
                                          b.get_element_as_complex(static_cast<uint64_t>(i));
        re += prod.real();
        im += prod.imag();
    }
    return {re, im};
}

std::unique_ptr<VectorBase> cross_product(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::invalid_argument("Cross product only defined for 3D vectors");
    }
    
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* b_dbl = dynamic_cast<const DenseVector<double>*>(&b);
    
    double ax, ay, az, bx, by, bz;
    
    if (a_dbl) {
        const double* d = a_dbl->data();
        ax = d[0]; ay = d[1]; az = d[2];
    } else {
        ax = a.get_element_as_double(0);
        ay = a.get_element_as_double(1);
        az = a.get_element_as_double(2);
    }
    
    if (b_dbl) {
        const double* d = b_dbl->data();
        bx = d[0]; by = d[1]; bz = d[2];
    } else {
        bx = b.get_element_as_double(0);
        by = b.get_element_as_double(1);
        bz = b.get_element_as_double(2);
    }
    
    auto result = std::make_unique<DenseVector<double>>(3, result_file);
    double* r = result->data();
    
    r[0] = ay*bz - az*by;
    r[1] = az*bx - ax*bz;
    r[2] = ax*by - ay*bx;
    
    return result;
}

std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, double scalar, const std::string& result_file) {
    const auto dt = v.get_data_type();
    if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
        const uint64_t n = v.size();
        std::unique_ptr<VectorBase> result;
        if (dt == DataType::COMPLEX_FLOAT16) {
            result = std::make_unique<ComplexFloat16Vector>(n, result_file);
            auto* out = static_cast<ComplexFloat16Vector*>(result.get());
            auto* rdst = out->real_data();
            auto* idst = out->imag_data();
            ParallelFor(0, n, [&](size_t i) {
                const std::complex<double> z = v.get_element_as_complex(i) * scalar;
                rdst[i] = float16_t(z.real());
                idst[i] = float16_t(z.imag());
            });
        } else if (dt == DataType::COMPLEX_FLOAT32) {
            result = std::make_unique<DenseVector<std::complex<float>>>(n, result_file);
            auto* out = static_cast<DenseVector<std::complex<float>>*>(result.get());
            auto* dst = out->data();
            ParallelFor(0, n, [&](size_t i) {
                const std::complex<double> z = v.get_element_as_complex(i) * scalar;
                dst[i] = std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
            });
        } else {
            result = std::make_unique<DenseVector<std::complex<double>>>(n, result_file);
            auto* out = static_cast<DenseVector<std::complex<double>>*>(result.get());
            auto* dst = out->data();
            ParallelFor(0, n, [&](size_t i) {
                dst[i] = v.get_element_as_complex(i) * scalar;
            });
        }

        result->set_seed(v.get_seed());
        result->set_transposed(v.is_transposed());
        result->set_scalar(1.0);
        return result;
    }
    uint64_t n = v.size();
    std::unique_ptr<VectorBase> result;
    switch (v.get_data_type()) {
        case DataType::FLOAT16:
            result = std::make_unique<DenseVector<float16_t>>(n, result_file);
            break;
        case DataType::FLOAT32:
            result = std::make_unique<DenseVector<float>>(n, result_file);
            break;
        default:
            result = std::make_unique<DenseVector<double>>(n, result_file);
            break;
    }
    ComputeContext::instance().get_device()->scalar_multiply_vector(v, scalar, *result);
    return result;
}

std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, std::complex<double> scalar, const std::string& result_file) {
    const auto dt = v.get_data_type();
    if (dt != DataType::COMPLEX_FLOAT16 && dt != DataType::COMPLEX_FLOAT32 && dt != DataType::COMPLEX_FLOAT64) {
        throw std::invalid_argument("scalar_multiply_vector: complex scalar multiply requires a complex vector dtype");
    }

    const uint64_t n = v.size();
    std::unique_ptr<VectorBase> result;
    if (dt == DataType::COMPLEX_FLOAT16) {
        result = std::make_unique<ComplexFloat16Vector>(n, result_file);
        auto* out = static_cast<ComplexFloat16Vector*>(result.get());
        auto* rdst = out->real_data();
        auto* idst = out->imag_data();
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> z = v.get_element_as_complex(i) * scalar;
            rdst[i] = float16_t(z.real());
            idst[i] = float16_t(z.imag());
        });
    } else if (dt == DataType::COMPLEX_FLOAT32) {
        result = std::make_unique<DenseVector<std::complex<float>>>(n, result_file);
        auto* out = static_cast<DenseVector<std::complex<float>>*>(result.get());
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> z = v.get_element_as_complex(i) * scalar;
            dst[i] = std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
        });
    } else {
        result = std::make_unique<DenseVector<std::complex<double>>>(n, result_file);
        auto* out = static_cast<DenseVector<std::complex<double>>*>(result.get());
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            dst[i] = v.get_element_as_complex(i) * scalar;
        });
    }

    result->set_seed(v.get_seed());
    result->set_transposed(v.is_transposed());
    result->set_scalar(1.0);
    return result;
}

std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, int64_t scalar, const std::string& result_file) {
    const auto dt = v.get_data_type();
    if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
        return scalar_multiply_vector(v, static_cast<double>(scalar), result_file);
    }
    uint64_t n = v.size();
    // If vector is int, result is int. If vector is float, result is float.
    if (v.get_data_type() == DataType::INT32 || v.get_data_type() == DataType::BIT) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_multiply_vector(v, static_cast<double>(scalar), *result);
        return result;
    } else if (v.get_data_type() == DataType::INT16) {
        auto result = std::make_unique<DenseVector<int16_t>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_multiply_vector(v, static_cast<double>(scalar), *result);
        return result;
    } else if (v.get_data_type() == DataType::FLOAT16) {
        auto result = std::make_unique<DenseVector<float16_t>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_multiply_vector(v, static_cast<double>(scalar), *result);
        return result;
    } else if (v.get_data_type() == DataType::FLOAT32) {
        auto result = std::make_unique<DenseVector<float>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_multiply_vector(v, static_cast<double>(scalar), *result);
        return result;
    } else {
        auto result = std::make_unique<DenseVector<double>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_multiply_vector(v, static_cast<double>(scalar), *result);
        return result;
    }
}

std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, double scalar, const std::string& result_file) {
    const auto dt = v.get_data_type();
    if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
        return v.add_scalar(scalar, result_file);
    }
    uint64_t n = v.size();
    std::unique_ptr<VectorBase> result;
    switch (v.get_data_type()) {
        case DataType::FLOAT16:
            result = std::make_unique<DenseVector<float16_t>>(n, result_file);
            break;
        case DataType::FLOAT32:
            result = std::make_unique<DenseVector<float>>(n, result_file);
            break;
        default:
            result = std::make_unique<DenseVector<double>>(n, result_file);
            break;
    }
    ComputeContext::instance().get_device()->scalar_add_vector(v, scalar, *result);
    return result;
}

std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, int64_t scalar, const std::string& result_file) {
    const auto dt = v.get_data_type();
    if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
        return v.add_scalar(static_cast<double>(scalar), result_file);
    }
    uint64_t n = v.size();
    if (v.get_data_type() == DataType::INT32 || v.get_data_type() == DataType::BIT) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_add_vector(v, static_cast<double>(scalar), *result);
        return result;
    } else if (v.get_data_type() == DataType::INT16) {
        auto result = std::make_unique<DenseVector<int16_t>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_add_vector(v, static_cast<double>(scalar), *result);
        return result;
    } else if (v.get_data_type() == DataType::FLOAT16) {
        auto result = std::make_unique<DenseVector<float16_t>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_add_vector(v, static_cast<double>(scalar), *result);
        return result;
    } else if (v.get_data_type() == DataType::FLOAT32) {
        auto result = std::make_unique<DenseVector<float>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_add_vector(v, static_cast<double>(scalar), *result);
        return result;
    } else {
        auto result = std::make_unique<DenseVector<double>>(n, result_file);
        ComputeContext::instance().get_device()->scalar_add_vector(v, static_cast<double>(scalar), *result);
        return result;
    }
}

// --- Matrix-Vector Operations ---

std::unique_ptr<MatrixBase> outer_product(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("outer_product: vectors must be same size (square matrices only)");
    }
    uint64_t n = a.size();
    
    // Determine result type
    DataType res_dtype = promotion::resolve(promotion::BinaryOp::OuterProduct, a.get_data_type(), b.get_data_type()).result_dtype;
    
    auto result = ObjectFactory::create_matrix(n, res_dtype, MatrixType::DENSE_FLOAT, result_file);
    
    ComputeContext::instance().get_device()->outer_product(a, b, *result);
    
    return result;
}

std::unique_ptr<VectorBase> matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, const std::string& result_file) {
    if (m.cols() != v.size()) {
        throw std::invalid_argument("Matrix columns must match vector size");
    }
    
    uint64_t n = m.rows();
    DataType res_dtype = promotion::resolve(
        promotion::BinaryOp::MatrixVectorMultiply,
        m.get_data_type(),
        v.get_data_type()).result_dtype;
    
    std::unique_ptr<VectorBase> result;
    if (res_dtype == DataType::BIT) {
        result = std::make_unique<DenseVector<bool>>(n, result_file);
    } else if (res_dtype == DataType::INT8) {
        result = std::make_unique<DenseVector<int8_t>>(n, result_file);
    } else if (res_dtype == DataType::INT16) {
        result = std::make_unique<DenseVector<int16_t>>(n, result_file);
    } else if (res_dtype == DataType::INT32) {
        result = std::make_unique<DenseVector<int32_t>>(n, result_file);
    } else if (res_dtype == DataType::INT64) {
        result = std::make_unique<DenseVector<int64_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT8) {
        result = std::make_unique<DenseVector<uint8_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT16) {
        result = std::make_unique<DenseVector<uint16_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT32) {
        result = std::make_unique<DenseVector<uint32_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT64) {
        result = std::make_unique<DenseVector<uint64_t>>(n, result_file);
    } else if (res_dtype == DataType::FLOAT16) {
        result = std::make_unique<DenseVector<float16_t>>(n, result_file);
    } else if (res_dtype == DataType::FLOAT32) {
        result = std::make_unique<DenseVector<float>>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT16) {
        result = std::make_unique<ComplexFloat16Vector>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT32) {
        result = std::make_unique<DenseVector<std::complex<float>>>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT64) {
        result = std::make_unique<DenseVector<std::complex<double>>>(n, result_file);
    } else {
        result = std::make_unique<DenseVector<double>>(n, result_file);
    }
    
    ComputeContext::instance().get_device()->matrix_vector_multiply(m, v, *result);
    
    return result;
}

std::unique_ptr<VectorBase> vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, const std::string& result_file) {
    if (v.size() != m.rows()) {
        throw std::invalid_argument("Vector size must match matrix rows");
    }
    
    uint64_t n = m.cols();
    DataType res_dtype = promotion::resolve(
        promotion::BinaryOp::VectorMatrixMultiply,
        m.get_data_type(),
        v.get_data_type()).result_dtype;
    
    std::unique_ptr<VectorBase> result;
    if (res_dtype == DataType::BIT) {
        result = std::make_unique<DenseVector<bool>>(n, result_file);
    } else if (res_dtype == DataType::INT8) {
        result = std::make_unique<DenseVector<int8_t>>(n, result_file);
    } else if (res_dtype == DataType::INT16) {
        result = std::make_unique<DenseVector<int16_t>>(n, result_file);
    } else if (res_dtype == DataType::INT32) {
        result = std::make_unique<DenseVector<int32_t>>(n, result_file);
    } else if (res_dtype == DataType::INT64) {
        result = std::make_unique<DenseVector<int64_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT8) {
        result = std::make_unique<DenseVector<uint8_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT16) {
        result = std::make_unique<DenseVector<uint16_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT32) {
        result = std::make_unique<DenseVector<uint32_t>>(n, result_file);
    } else if (res_dtype == DataType::UINT64) {
        result = std::make_unique<DenseVector<uint64_t>>(n, result_file);
    } else if (res_dtype == DataType::FLOAT16) {
        result = std::make_unique<DenseVector<float16_t>>(n, result_file);
    } else if (res_dtype == DataType::FLOAT32) {
        result = std::make_unique<DenseVector<float>>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT16) {
        result = std::make_unique<ComplexFloat16Vector>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT32) {
        result = std::make_unique<DenseVector<std::complex<float>>>(n, result_file);
    } else if (res_dtype == DataType::COMPLEX_FLOAT64) {
        result = std::make_unique<DenseVector<std::complex<double>>>(n, result_file);
    } else {
        result = std::make_unique<DenseVector<double>>(n, result_file);
    }
    
    // Result of v * M is a row vector
    result->set_transposed(true);

    ComputeContext::instance().get_device()->vector_matrix_multiply(v, m, *result);
    
    return result;
}

using Float32Matrix = DenseMatrix<float>;
using Float16Matrix2 = DenseMatrix<float16_t>;
using TriangularBitMatrix = TriangularMatrix<bool>;
using DenseBitMatrix = DenseMatrix<bool>;

std::unique_ptr<MatrixBase> dispatch_matmul(const MatrixBase& a, const MatrixBase& b, std::string saveas) {
    if (saveas.empty()) saveas = make_unique_storage_file("matmul");

    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    // Try to cast to known types
    auto* a_fm = dynamic_cast<const FloatMatrix*>(&a);
    auto* b_fm = dynamic_cast<const FloatMatrix*>(&b);
    auto* a_fm32 = dynamic_cast<const Float32Matrix*>(&a);
    auto* b_fm32 = dynamic_cast<const Float32Matrix*>(&b);
    auto* a_fm16 = dynamic_cast<const Float16Matrix2*>(&a);
    auto* b_fm16 = dynamic_cast<const Float16Matrix2*>(&b);

    bool a_is_id = (a.get_matrix_type() == MatrixType::IDENTITY);
    bool b_is_id = (b.get_matrix_type() == MatrixType::IDENTITY);

    // Identity x Identity -> Identity
    if (a_is_id && b_is_id) {
        DataType res_dtype = promotion::resolve(
            promotion::BinaryOp::Matmul,
            a.get_data_type(),
            b.get_data_type()).result_dtype;
        auto res = ObjectFactory::create_matrix(a.rows(), b.cols(), res_dtype, MatrixType::IDENTITY, saveas);
        res->set_scalar(a.get_scalar() * b.get_scalar());
        return res;
    }

    // Identity x Any -> Any * scalar
    if (a_is_id && a.rows() == a.cols()) {
        std::complex<double> s = a.get_scalar();
        if (s.imag() == 0.0) {
            return b.multiply_scalar(s.real(), saveas);
        }
        return b.multiply_scalar(s, saveas);
    }

    // Any x Identity -> Any * scalar
    if (b_is_id && b.rows() == b.cols()) {
        std::complex<double> s = b.get_scalar();
        if (s.imag() == 0.0) {
            return a.multiply_scalar(s.real(), saveas);
        }
        return a.multiply_scalar(s, saveas);
    }

    // Complex matmul (CPU fallback): allocate promoted complex dtype and dispatch via device.
    const bool a_is_cplx =
        (a.get_data_type() == DataType::COMPLEX_FLOAT16 || a.get_data_type() == DataType::COMPLEX_FLOAT32 ||
         a.get_data_type() == DataType::COMPLEX_FLOAT64);
    const bool b_is_cplx =
        (b.get_data_type() == DataType::COMPLEX_FLOAT16 || b.get_data_type() == DataType::COMPLEX_FLOAT32 ||
         b.get_data_type() == DataType::COMPLEX_FLOAT64);
    if (a_is_cplx || b_is_cplx) {
        if (a.cols() != b.rows()) throw std::invalid_argument("Dimension mismatch");
        const DataType res_dtype = promotion::resolve(
            promotion::BinaryOp::Matmul,
            a.get_data_type(),
            b.get_data_type()).result_dtype;
        auto res = ObjectFactory::create_matrix(a.rows(), b.cols(), res_dtype, MatrixType::DENSE_FLOAT, saveas);
        ComputeContext::instance().get_device()->matmul(a, b, *res);
        return res;
    }

    // Use ComputeDevice for FloatMatrix x FloatMatrix
    if (a_fm && b_fm && ComputeContext::instance().is_gpu_active()) {
        auto res = std::make_unique<FloatMatrix>(a.rows(), b.cols(), saveas);
        ComputeContext::instance().get_device()->matmul(*a_fm, *b_fm, *res);
        return res;
    }

    // Use ComputeDevice for Float32Matrix x Float32Matrix
    if (a_fm32 && b_fm32 && ComputeContext::instance().is_gpu_active()) {
        auto res = std::make_unique<Float32Matrix>(a.rows(), b.cols(), saveas);
        ComputeContext::instance().get_device()->matmul(*a_fm32, *b_fm32, *res);
        return res;
    }

    // Mixed float precision underpromotes to float32 (CPU fallback via AutoSolver)
    if ((a_fm && b_fm32) || (a_fm32 && b_fm)) {
        auto res = std::make_unique<Float32Matrix>(a.rows(), b.cols(), saveas);
        ComputeContext::instance().get_device()->matmul(a, b, *res);
        return res;
    }

    // Any float16 participation: allocate promoted dtype and dispatch via device.
    // Use dtype checks (not just dynamic_cast) so this also works for non-dense float16 views.
    const bool a_is_float =
        (a.get_data_type() == DataType::FLOAT16 || a.get_data_type() == DataType::FLOAT32 ||
         a.get_data_type() == DataType::FLOAT64);
    const bool b_is_float =
        (b.get_data_type() == DataType::FLOAT16 || b.get_data_type() == DataType::FLOAT32 ||
         b.get_data_type() == DataType::FLOAT64);
    if (a_is_float && b_is_float &&
        (a.get_data_type() == DataType::FLOAT16 || b.get_data_type() == DataType::FLOAT16)) {
        const DataType res_dtype = promotion::resolve(
            promotion::BinaryOp::Matmul,
            a.get_data_type(),
            b.get_data_type()).result_dtype;
        auto res = ObjectFactory::create_matrix(a.rows(), b.cols(), res_dtype, MatrixType::DENSE_FLOAT, saveas);
        ComputeContext::instance().get_device()->matmul(a, b, *res);
        return res;
    }

    // Fallback to existing logic for other types
    if (a_fm && b_fm) {
         return a_fm->multiply(*b_fm, saveas);
    }
    if (a_fm32 && b_fm32) {
         return a_fm32->multiply(*b_fm32, saveas);
    }
    if (a_fm16 && b_fm16) {
        auto res = std::make_unique<Float16Matrix>(a.rows(), b.cols(), saveas);
        ComputeContext::instance().get_device()->matmul(*a_fm16, *b_fm16, *res);
        return res;
    }

    auto* a_im = dynamic_cast<const IntegerMatrix*>(&a);
    auto* a_tbm = dynamic_cast<const TriangularBitMatrix*>(&a);
    auto* a_dbm = dynamic_cast<const DenseBitMatrix*>(&a);
    auto* a_i16m = dynamic_cast<const Int16Matrix*>(&a);
    
    auto* b_im = dynamic_cast<const IntegerMatrix*>(&b);
    auto* b_tbm = dynamic_cast<const TriangularBitMatrix*>(&b);
    auto* b_dbm = dynamic_cast<const DenseBitMatrix*>(&b);
    auto* b_i16m = dynamic_cast<const Int16Matrix*>(&b);

    // DenseBitMatrix x {IntegerMatrix, FloatMatrix, Float32Matrix}
    // Scale-first: keep A bit-packed and avoid materializing it to dense ints/floats.
    if (a_dbm && b_im) {
        auto res = std::make_unique<IntegerMatrix>(a.rows(), b.cols(), saveas);
        ComputeContext::instance().get_device()->matmul(*a_dbm, *b_im, *res);
        return res;
    }
    if (a_dbm && b_i16m) {
        auto res = std::make_unique<Int16Matrix>(a.rows(), b.cols(), saveas);
        ComputeContext::instance().get_device()->matmul(*a_dbm, *b_i16m, *res);
        return res;
    }
    if (a_dbm && b_fm) {
        auto res = std::make_unique<FloatMatrix>(a.rows(), b.cols(), saveas);
        ComputeContext::instance().get_device()->matmul(*a_dbm, *b_fm, *res);
        return res;
    }
    if (a_dbm && b_fm32) {
        auto res = std::make_unique<Float32Matrix>(a.rows(), b.cols(), saveas);
        ComputeContext::instance().get_device()->matmul(*a_dbm, *b_fm32, *res);
        return res;
    }

    // IntegerMatrix x IntegerMatrix
    if (a_im && b_im) return a_im->multiply(*b_im, saveas);

    // Int16Matrix x Int16Matrix
    if (a_i16m && b_i16m) return a_i16m->multiply(*b_i16m, saveas);

    // Mixed int16/int32 -> allocate promoted dtype and dispatch via device
    if ((a_i16m && b_im) || (a_im && b_i16m)) {
        const DataType res_dtype = promotion::resolve(
            promotion::BinaryOp::Matmul,
            a.get_data_type(),
            b.get_data_type()).result_dtype;
        auto res = ObjectFactory::create_matrix(a.rows(), b.cols(), res_dtype, MatrixType::DENSE_FLOAT, saveas);
        ComputeContext::instance().get_device()->matmul(a, b, *res);
        return res;
    }

    // Same-dtype integer matmul beyond int16/int32 (int8/int64/uint*): allocate and dispatch via device.
    const auto a_dt = a.get_data_type();
    const auto b_dt = b.get_data_type();
    const bool a_is_int =
        (a_dt == DataType::INT8 || a_dt == DataType::INT16 || a_dt == DataType::INT32 || a_dt == DataType::INT64 ||
         a_dt == DataType::UINT8 || a_dt == DataType::UINT16 || a_dt == DataType::UINT32 || a_dt == DataType::UINT64);
    const bool b_is_int =
        (b_dt == DataType::INT8 || b_dt == DataType::INT16 || b_dt == DataType::INT32 || b_dt == DataType::INT64 ||
         b_dt == DataType::UINT8 || b_dt == DataType::UINT16 || b_dt == DataType::UINT32 || b_dt == DataType::UINT64);
    if (a_is_int && b_is_int && a_dt == b_dt) {
        const DataType res_dtype = promotion::resolve(
            promotion::BinaryOp::Matmul,
            a_dt,
            b_dt).result_dtype;
        auto res = ObjectFactory::create_matrix(a.rows(), b.cols(), res_dtype, MatrixType::DENSE_FLOAT, saveas);
        ComputeContext::instance().get_device()->matmul(a, b, *res);
        return res;
    }

    // DenseBitMatrix x DenseBitMatrix
    if (a_dbm && b_dbm) return a_dbm->multiply(*b_dbm, saveas);

    // TriangularBitMatrix x TriangularBitMatrix -> TriangularIntegerMatrix
    if (a_tbm && b_tbm) {
        // GPU Acceleration for BitMatrices is disabled by default due to overhead.
        // Users must explicitly convert to FloatMatrix/Float32Matrix to use GPU.
        return a_tbm->multiply(*b_tbm, saveas);
    }

    throw std::runtime_error("Unsupported matrix multiplication types.");
}

}

// --- Special Solvers (Implementation from MatrixOperations.cpp) ---

std::unique_ptr<pycauset::TriangularMatrix<double>> compute_k_matrix(
    const pycauset::TriangularMatrix<bool>& C, 
    double a, 
    const std::string& output_path, 
    int num_threads
) {
    uint64_t n = C.rows();
    auto K = std::make_unique<pycauset::TriangularMatrix<double>>(n, output_path);
    
    // Get base pointer for C
    const char* c_raw_bytes = reinterpret_cast<const char*>(C.data());
    char* k_base_ptr = reinterpret_cast<char*>(K->data());

    pycauset::ParallelFor(0, n, [&](size_t j) {
        // Allocate temporary column buffer for K[0...j-1][j]
        std::vector<double> col_j(j + 1, 0.0);
        
        // Iterate i from j-1 down to 0
        for (int64_t i = j - 1; i >= 0; --i) {
            double sum = 0.0;
            
            uint64_t row_offset = C.get_row_offset(i);
            const uint64_t* row_ptr = reinterpret_cast<const uint64_t*>(c_raw_bytes + row_offset);
            
            if (j > i + 1) {
                uint64_t max_bit_index = j - i - 2;
                uint64_t num_words = (max_bit_index / 64) + 1;
                
                for (uint64_t w = 0; w < num_words; ++w) {
                    uint64_t word = row_ptr[w];
                    if (word == 0) continue;
                    
                    if (w == num_words - 1) {
                        uint64_t bits_in_last_word = (max_bit_index % 64) + 1;
                        if (bits_in_last_word < 64) {
                            uint64_t mask = (1ULL << bits_in_last_word) - 1;
                            word &= mask;
                        }
                    }

                    while (word != 0) {
                        int bit = std::countr_zero(word);
                        uint64_t m = (i + 1) + (w * 64 + bit);
                        
                        if (m < j) {
                            sum += col_j[m];
                        }
                        word &= (word - 1);
                    }
                }
            }
            
            bool c_ij = false;
            uint64_t bit_offset_j = j - (i + 1);
            uint64_t word_idx_j = bit_offset_j / 64;
            uint64_t bit_idx_j = bit_offset_j % 64;
            
            uint64_t word_j = row_ptr[word_idx_j];
            if ((word_j >> bit_idx_j) & 1ULL) {
                c_ij = true;
            }
            
            double val = ( (c_ij ? 1.0 : 0.0) - sum ) / a;
            col_j[i] = val;
            
            uint64_t k_row_offset = K->get_row_offset(i);
            uint64_t k_col_idx = j - (i + 1);
            double* k_row_ptr = reinterpret_cast<double*>(k_base_ptr + k_row_offset);
            k_row_ptr[k_col_idx] = val;
        }
    });
    return K;
}
