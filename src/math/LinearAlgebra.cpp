#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include <utility>
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/BitClosureMatrix.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/StructureResolver.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/compute/ComputeDevice.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <functional>
#include <iostream>
#include <complex>

namespace pycauset {

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

bool is_identity(const MatrixBase& m) {
    return m.get_structure() == Structure::IDENTITY;
}

// --- Matrix Operations ---

// C + C^T / C - C^T for a causal (bit, strictly upper triangular) C: a closure view over C's
// bits (LAZY.md L2b). sign_op = +1 for add, -1 for subtract.
static std::unique_ptr<MatrixBase> try_bit_closure(const MatrixBase& a, const MatrixBase& b, int sign_op,
                                                   const std::string& result_file) {
    if (!lazy_enabled().load() || !result_file.empty()) return nullptr;
    if (a.get_data_type() != DataType::BIT || b.get_data_type() != DataType::BIT) return nullptr;
    if (a.get_structure() != Structure::TRIANGULAR || b.get_structure() != Structure::TRIANGULAR) return nullptr;
    if (!a.shared_mapper() || a.shared_mapper() != b.shared_mapper()) return nullptr;
    if (a.is_transposed() == b.is_transposed()) return nullptr;
    if (a.get_scalar() != std::complex<double>(1.0, 0.0) || b.get_scalar() != std::complex<double>(1.0, 0.0)) return nullptr;
    if (a.is_subview() || b.is_subview()) return nullptr;
    if (promotion::resolve(sign_op > 0 ? promotion::BinaryOp::Add : promotion::BinaryOp::Subtract,
                           DataType::BIT, DataType::BIT).result_dtype != DataType::INT32) return nullptr;
    auto out = std::make_unique<BitClosureMatrix>(a.rows(), a.shared_mapper(), sign_op);
    if (sign_op < 0 && a.is_transposed()) out->set_scalar(-1.0);  // C^T - C = -(C - C^T)
    return out;
}

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

    if (auto closure = try_bit_closure(a, b, +1, result_file)) return closure;
    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "add");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();

    DataType res_dtype = promotion::resolve(promotion::BinaryOp::Add, type_a, type_b).result_dtype;
    Structure res_mtype = structure::resolve_result_structure(promotion::BinaryOp::Add, a, b, res_dtype);

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

    if (auto closure = try_bit_closure(a, b, -1, result_file)) return closure;
    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "subtract");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();

    DataType res_dtype = promotion::resolve(promotion::BinaryOp::Subtract, type_a, type_b).result_dtype;
    Structure res_mtype = structure::resolve_result_structure(promotion::BinaryOp::Subtract, a, b, res_dtype);

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

    DataType res_dtype = promotion::resolve(promotion::BinaryOp::ElementwiseMultiply, type_a, type_b).result_dtype;
    Structure res_mtype = structure::resolve_result_structure(promotion::BinaryOp::ElementwiseMultiply, a, b, res_dtype);

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
    Structure res_mtype = Structure::DENSE;

    auto result = ObjectFactory::create_matrix(out_rows, out_cols, res_dtype, res_mtype, result_file);

    ComputeContext::instance().get_device()->elementwise_divide(a, b, *result);

    return result;
}

// --- Vector Operations ---



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
    return ComputeContext::instance().get_device()->dot_complex(a, b);
}

std::unique_ptr<VectorBase> cross_product(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::invalid_argument("Cross product only defined for 3D vectors");
    }

    auto result = std::make_unique<DenseVector<double>>(3, result_file);
    ComputeContext::instance().get_device()->cross_product(a, b, *result);
    return result;
}






// --- Matrix-Vector Operations ---

std::unique_ptr<MatrixBase> outer_product(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("outer_product: vectors must be same size (square matrices only)");
    }
    uint64_t n = a.size();
    
    // Determine result type
    DataType res_dtype = promotion::resolve(promotion::BinaryOp::OuterProduct, a.get_data_type(), b.get_data_type()).result_dtype;
    
    auto result = ObjectFactory::create_matrix(n, res_dtype, Structure::DENSE, result_file);
    
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
        result = std::make_unique<DenseVector<pycauset::float16_t>>(n, result_file);
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
        result = std::make_unique<DenseVector<pycauset::float16_t>>(n, result_file);
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


namespace {
bool mm_complex(DataType d) {
    return d == DataType::COMPLEX_FLOAT16 || d == DataType::COMPLEX_FLOAT32 || d == DataType::COMPLEX_FLOAT64;
}
bool mm_int(DataType d) { return is_integer_or_bit(d) && d != DataType::BIT; }

// The kernels' own scalar handling: the factor of a float operand is folded into the result
// scalar; integer kernels take factor 1 (an integer view factor is written out, so overflow
// and wrapping are decided on the values NumPy sees).
bool mm_factor_ok(const MatrixBase& m) {
    const std::complex<double> s = m.get_scalar();
    const DataType d = m.get_data_type();
    if (mm_complex(d)) return !m.is_conjugated();
    if (mm_int(d) || d == DataType::BIT) return s == std::complex<double>(1.0, 0.0);
    return s.imag() == 0.0;
}

// Dense kernel contract (CpuSolver::matmul): DenseMatrix<c>, no window, factor as above.
bool mm_dense_ready(const MatrixBase& m, DataType c) {
    return m.get_structure() == Structure::DENSE && m.get_data_type() == c && !m.is_subview() && mm_factor_ok(m);
}

// Triangular kernel contract: upper TriangularMatrix<r>, no window, factor as above.
bool mm_triangular_ready(const MatrixBase& m, DataType r) {
    return m.get_structure() == Structure::TRIANGULAR && m.get_data_type() == r && !m.is_subview() &&
           !m.is_transposed() && mm_factor_ok(m);
}

bool mm_has_triangular_kernel(DataType r) {
    return r == DataType::FLOAT64 || r == DataType::FLOAT32 || (mm_int(r));
}

std::pair<double, double> mm_int_range(DataType r) {
    switch (r) {
        case DataType::INT8: return {-128.0, 127.0};
        case DataType::INT16: return {-32768.0, 32767.0};
        case DataType::INT32: return {-2147483648.0, 2147483647.0};
        case DataType::INT64: return {-9223372036854775808.0, 9223372036854775807.0};
        case DataType::UINT8: return {0.0, 255.0};
        case DataType::UINT16: return {0.0, 65535.0};
        case DataType::UINT32: return {0.0, 4294967295.0};
        case DataType::UINT64: return {0.0, 18446744073709551615.0};
        default: throw std::logic_error("mm_int_range: not an integer dtype");
    }
}

bool mm_has_diagonal_class(DataType r) { return r == DataType::FLOAT64 || mm_int(r); }

template <typename T>
std::unique_ptr<MatrixBase> mm_make_triangular_t(uint64_t n, const std::string& saveas, bool diag) {
    return std::make_unique<TriangularMatrix<T>>(n, saveas, diag);
}

std::unique_ptr<MatrixBase> mm_make_triangular(DataType r, uint64_t n, const std::string& saveas, bool diag) {
    switch (r) {
        case DataType::FLOAT64: return mm_make_triangular_t<double>(n, saveas, diag);
        case DataType::FLOAT32: return mm_make_triangular_t<float>(n, saveas, diag);
        case DataType::INT8: return mm_make_triangular_t<int8_t>(n, saveas, diag);
        case DataType::INT16: return mm_make_triangular_t<int16_t>(n, saveas, diag);
        case DataType::INT32: return mm_make_triangular_t<int32_t>(n, saveas, diag);
        case DataType::INT64: return mm_make_triangular_t<int64_t>(n, saveas, diag);
        case DataType::UINT8: return mm_make_triangular_t<uint8_t>(n, saveas, diag);
        case DataType::UINT16: return mm_make_triangular_t<uint16_t>(n, saveas, diag);
        case DataType::UINT32: return mm_make_triangular_t<uint32_t>(n, saveas, diag);
        case DataType::UINT64: return mm_make_triangular_t<uint64_t>(n, saveas, diag);
        default: throw std::logic_error("mm_make_triangular: no triangular kernel dtype");
    }
}

bool mm_has_diagonal(const MatrixBase& m) {
    auto* t = dynamic_cast<const TriangularMatrixBase*>(&m);
    return t != nullptr && t->has_diagonal();
}
}  // namespace

// Matrix product. One normaliser in front of a few kernels with declared contracts
// (CpuSolver::matmul); review/PLAN.md Phase N.1. The result dtype is the resolver's. Rules:
//   R1 identity:   I @ X and X @ I are scalar operations on X.
//   R2 bit-packed: bit @ bit counts paths (int32; strictly upper x strictly upper stays
//                  triangular); bit @ dense float64/float32/int32/int16 keeps the bit operand packed.
//   R3 triangular: upper @ upper stays upper triangular (every real dtype with a kernel).
//   R4 diagonal:   diag @ diag stays diagonal (O(n)).
//   R5 dense:      everything else; operands become DenseMatrix<c> (c = r, or float32 /
//                  complex64 for the float16 dtypes, which have no arithmetic) and the result
//                  is cast to r.
// Structured rules skip structural zeros, so they differ from NumPy only where a skipped
// 0 * inf or 0 * nan would have produced nan.
std::unique_ptr<MatrixBase> dispatch_matmul(const MatrixBase& a, const MatrixBase& b, std::string saveas) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Dimension mismatch");
    }
    const DataType da = a.get_data_type(), db = b.get_data_type();
    const DataType r = promotion::resolve(promotion::BinaryOp::Matmul, da, db).result_dtype;
    const Structure sa = a.get_structure(), sb = b.get_structure();
    ComputeDevice* device = ComputeContext::instance().get_device();
    const uint64_t m = a.rows(), n = b.cols();

    // R1 identity.
    if (sa == Structure::IDENTITY && sb == Structure::IDENTITY) {
        // An identity entry's value is conj?(1 * s) (math/ViewValue.hpp).
        auto res = ObjectFactory::create_matrix(m, n, r, Structure::IDENTITY, saveas);
        res->set_scalar(view_value(1.0, a.get_scalar(), a.is_conjugated()) *
                        view_value(1.0, b.get_scalar(), b.is_conjugated()));
        return res;
    }
    auto identity_times = [&](const MatrixBase& id, const MatrixBase& other) {
        const std::complex<double> s = view_value(1.0, id.get_scalar(), id.is_conjugated());
        const ScalarArg arg = s.imag() == 0.0 ? ScalarArg::from_double(s.real()) : ScalarArg::from_complex(s);
        auto out = scalar_op(other, ScalarOpKind::Mul, arg);  // integer-safe (F-001), promotes
        if (out->get_data_type() != r) out = cast_to(*out, r);
        return out;
    };
    if (sa == Structure::IDENTITY && a.rows() == a.cols()) return identity_times(a, b);
    if (sb == Structure::IDENTITY && b.rows() == b.cols()) return identity_times(b, a);

    // R2 bit-packed.
    auto bit_ready = [](const MatrixBase& x) {
        return x.get_data_type() == DataType::BIT && !x.is_subview() && mm_factor_ok(x) &&
               (x.get_structure() == Structure::DENSE || x.get_structure() == Structure::TRIANGULAR);
    };
    if (r == DataType::INT32 && bit_ready(a) && bit_ready(b)) {
        if (sa == Structure::TRIANGULAR && sb == Structure::TRIANGULAR && !a.is_transposed() && !b.is_transposed()) {
            debug_trace::set_last("cpu.matmul.bit_triangular");
            return static_cast<const TriangularMatrix<bool>&>(a).multiply(static_cast<const TriangularMatrix<bool>&>(b),
                                                                          saveas);
        }
        std::unique_ptr<MatrixBase> ha = sa == Structure::DENSE ? nullptr : write_out(a);
        std::unique_ptr<MatrixBase> hb = sb == Structure::DENSE ? nullptr : write_out(b);
        auto res = std::make_unique<DenseMatrix<int32_t>>(m, n, saveas);
        device->matmul(ha ? *ha : a, hb ? *hb : b, *res);
        return res;
    }
    if (bit_ready(a) && sa == Structure::DENSE && mm_dense_ready(b, r) &&
        (r == DataType::FLOAT64 || r == DataType::FLOAT32 || r == DataType::INT32 || r == DataType::INT16)) {
        auto res = ObjectFactory::create_matrix(m, n, r, Structure::DENSE, saveas);
        device->matmul(a, b, *res);
        return res;
    }

    // R3 triangular. Operands are brought to TriangularMatrix<r> (cast keeps the structure);
    // a lower or windowed triangular operand is not upper triangular and goes dense.
    if (sa == Structure::TRIANGULAR && sb == Structure::TRIANGULAR && mm_has_triangular_kernel(r) &&
        !a.is_transposed() && !b.is_transposed() && !a.is_subview() && !b.is_subview()) {
        std::unique_ptr<MatrixBase> ha = mm_triangular_ready(a, r) ? nullptr : cast_to(a, r);
        std::unique_ptr<MatrixBase> hb = mm_triangular_ready(b, r) ? nullptr : cast_to(b, r);
        const MatrixBase& A = ha ? *ha : a;
        const MatrixBase& B = hb ? *hb : b;
        if (mm_triangular_ready(A, r) && mm_triangular_ready(B, r)) {
            // Upper x upper; the diagonal is a(i,i) b(i,i), stored only if both have one.
            auto res = mm_make_triangular(r, m, saveas, mm_has_diagonal(A) && mm_has_diagonal(B));
            device->matmul(A, B, *res);
            return res;
        }
    }

    // R4 diagonal. Entries are read through the view-aware accessor; integer results are
    // formed exactly only while every product stays below 2^53 (measured), otherwise dense.
    if (sa == Structure::DIAGONAL && sb == Structure::DIAGONAL && mm_has_diagonal_class(r) && !a.is_subview() &&
        !b.is_subview()) {
        std::vector<double> d(m);
        bool exact = true;
        for (uint64_t i = 0; i < m && exact; ++i) {
            const std::complex<double> x = a.get_element_as_complex(i, i) * b.get_element_as_complex(i, i);
            d[i] = x.real();
            if (mm_int(r) && !(std::abs(d[i]) < 9007199254740992.0)) exact = false;
        }
        if (exact) {
            if (mm_int(r)) {
                const auto [lo, hi] = mm_int_range(r);
                for (double v : d) {
                    if (v < lo || v > hi) {
                        throw std::overflow_error("Integer matmul overflow: result does not fit in output dtype");
                    }
                }
            }
            auto res = ObjectFactory::create_matrix(m, n, r, Structure::DIAGONAL, saveas);
            for (uint64_t i = 0; i < m; ++i) res->set_element_as_double(i, i, d[i]);
            return res;
        }
    }

    // R5 dense.
    const DataType c = r == DataType::FLOAT16 ? DataType::FLOAT32
                     : r == DataType::COMPLEX_FLOAT16 ? DataType::COMPLEX_FLOAT32
                     : r;
    std::unique_ptr<MatrixBase> ha = mm_dense_ready(a, c) ? nullptr : to_dense(a, c);
    std::unique_ptr<MatrixBase> hb = mm_dense_ready(b, c) ? nullptr : to_dense(b, c);
    auto res = ObjectFactory::create_matrix(m, n, c, Structure::DENSE, c == r ? saveas : std::string());
    device->matmul(ha ? *ha : a, hb ? *hb : b, *res);
    if (c != r) return cast_to(*res, r);
    return res;
}

std::unique_ptr<MatrixBase> cholesky(const MatrixBase& a, const std::string& result_file) {
    if (!is_plain(a)) return cholesky(*write_out(a), result_file);  // kernels read storage (F-062)
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("cholesky: matrix must be square");
    }

    const DataType dtype = a.get_data_type();
    if (dtype != DataType::FLOAT32 && dtype != DataType::FLOAT64) {
        throw std::runtime_error("cholesky: only float32/float64 are supported");
    }

    auto out = ObjectFactory::create_matrix(a.rows(), a.cols(), dtype, Structure::DENSE, result_file);
    ComputeContext::instance().get_device()->cholesky(a, *out);
    return out;
}

std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>> lu(const MatrixBase& a, const std::string& result_file)
{
    if (!is_plain(a)) return lu(*write_out(a), result_file);  // kernels read storage (F-062)
    // LU decomposition: A = P * L * U
    // For now we assume typical behavior where P, L, U are returned.
    // Dimensions: A is (M, N). P is (M, M), L is (M, K), U is (K, N) where K = min(M, N).
    // However, typical LAPACK getrf returns packed LU and permutation vector.
    // The device implementation will likely handle the specifics, but here we allocate space.
    // If wrapping standard LAPACK, we might return P as indices, but the python interface typically wants a matrix for P if requested, or just L, U.
    // Let's stick to returning Matrices for P, L, U to match the signature.
    
    // For simplicity in this binding layer, we allocate same-sized or appropriately sized matrices.
    // L and U are stored in 'a' usually for in-place, but here we want separate outputs.
    // We'll let the device implementation handle the complexity of unpacking if needed, or we just allocate standard sizes.
    // P: (M, M), L: (M, K), U: (K, N). K=min(M,N).
    
    uint64_t m = a.rows();
    uint64_t n = a.cols();
    uint64_t k = std::min(m, n);
    
    DataType dtype = a.get_data_type();
    
    auto p = ObjectFactory::create_matrix(m, m, dtype, Structure::DENSE, result_file.empty() ? "" : result_file + "_p");
    auto l = ObjectFactory::create_matrix(m, k, dtype, Structure::DENSE, result_file.empty() ? "" : result_file + "_l");
    auto u = ObjectFactory::create_matrix(k, n, dtype, Structure::DENSE, result_file.empty() ? "" : result_file + "_u");

    ComputeContext::instance().get_device()->lu(a, *p, *l, *u);
    
    return std::make_tuple(std::move(p), std::move(l), std::move(u));
}

std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>> qr(const MatrixBase& a, const std::string& result_file)
{
    if (!is_plain(a)) return qr(*write_out(a), result_file);  // kernels read storage (F-062)
    uint64_t m = a.rows();
    uint64_t n = a.cols();
    uint64_t k = std::min(m, n);
    
    DataType dtype = a.get_data_type();

    // Mode 'reduced' is typical default. Q is (M, K), R is (K, N).
    auto q = ObjectFactory::create_matrix(m, k, dtype, Structure::DENSE, result_file.empty() ? "" : result_file + "_q");
    auto r = ObjectFactory::create_matrix(k, n, dtype, Structure::DENSE, result_file.empty() ? "" : result_file + "_r");

    ComputeContext::instance().get_device()->qr(a, *q, *r);
    
    return std::make_tuple(std::move(q), std::move(r));
}

// Result dtype of solve/svd (LAPACK dtypes), NumPy's rule: integers and bit solve in float64, float16/float32 in
// float32, complex stays complex (complex64 only when nothing needs double precision).
static DataType solve_dtype(DataType a, DataType b) {
    auto cplx = [](DataType d) {
        return d == DataType::COMPLEX_FLOAT16 || d == DataType::COMPLEX_FLOAT32 || d == DataType::COMPLEX_FLOAT64;
    };
    auto single = [](DataType d) {
        return d == DataType::FLOAT16 || d == DataType::FLOAT32 || d == DataType::COMPLEX_FLOAT16 ||
               d == DataType::COMPLEX_FLOAT32;
    };
    const bool sp = single(a) && single(b);
    if (cplx(a) || cplx(b)) return sp ? DataType::COMPLEX_FLOAT32 : DataType::COMPLEX_FLOAT64;
    return sp ? DataType::FLOAT32 : DataType::FLOAT64;
}

std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> svd(
    const MatrixBase& a, bool full_matrices)
{
    // The LAPACK kernel reads dense storage in a LAPACK dtype: integers/bit in float64,
    // float16 in float32, complex32 in complex64 (NumPy's rule); any structure or view is
    // written out first (F-062, F-079: complex SVD used to fall back to NumPy).
    {
        const DataType in = a.get_data_type();
        const DataType want = solve_dtype(in, in);
        if (a.get_structure() != Structure::DENSE || in != want || !is_plain(a)) {
            return svd(*to_dense(a, want), full_matrices);
        }
    }
    const uint64_t m = a.rows(), n = a.cols(), k = std::min(m, n);
    const DataType dtype = a.get_data_type();
    const DataType s_dtype = dtype == DataType::COMPLEX_FLOAT32 ? DataType::FLOAT32
                           : dtype == DataType::COMPLEX_FLOAT64 ? DataType::FLOAT64 : dtype;
    // NumPy shapes: U (m, m) and VT (n, n) when full_matrices, else (m, k) and (k, n).
    auto u = ObjectFactory::create_matrix(m, full_matrices ? m : k, dtype, Structure::DENSE, "");
    auto s = ObjectFactory::create_vector(k, s_dtype, Structure::VECTOR, "");
    auto vt = ObjectFactory::create_matrix(full_matrices ? n : k, n, dtype, Structure::DENSE, "");
    ComputeContext::instance().get_device()->svd(a, *u, *s, *vt);
    return std::make_tuple(std::move(u), std::move(s), std::move(vt));
}

std::unique_ptr<MatrixBase> solve(const MatrixBase& a, const MatrixBase& b, const std::string& result_file)
{
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("solve: coefficient matrix 'a' must be square");
    }
    if (a.rows() != b.rows()) {
        throw std::invalid_argument("solve: 'b' must have the same number of rows as 'a'");
    }
    // The LAPACK kernel reads dense storage of one dtype: any structure, view or other dtype
    // is written out first (O(n^2) in front of O(n^3); every input used to fall back to NumPy
    // or read packed storage as dense, F-068).
    const DataType dt = solve_dtype(a.get_data_type(), b.get_data_type());
    auto ready = [dt](const MatrixBase& m) {
        return m.get_structure() == Structure::DENSE && m.get_data_type() == dt && is_plain(m);
    };
    std::unique_ptr<MatrixBase> pa = ready(a) ? nullptr : to_dense(a, dt);
    std::unique_ptr<MatrixBase> pb = ready(b) ? nullptr : to_dense(b, dt);
    const MatrixBase& A = pa ? *pa : a;
    const MatrixBase& B = pb ? *pb : b;

    auto x = ObjectFactory::create_matrix(B.rows(), B.cols(), dt, Structure::DENSE, result_file);
    ComputeContext::instance().get_device()->solve(A, B, *x);
    return x;
}

// Exact symmetry test for real matrices (no tolerance: a "nearly symmetric" matrix is
// treated as general, which is always correct, just slower).
static bool is_exactly_symmetric(const MatrixBase& a) {
    if (a.rows() != a.cols()) return false;
    const DataType dt = a.get_data_type();
    if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
        return false;
    }
    if (a.get_structure() == Structure::SYMMETRIC || a.get_structure() == Structure::IDENTITY ||
        a.get_structure() == Structure::DIAGONAL) {
        return true;
    }
    const uint64_t n = a.rows();
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = i + 1; j < n; ++j) {
            if (a.get_element_as_double(i, j) != a.get_element_as_double(j, i)) return false;
        }
    }
    return true;
}

std::unique_ptr<VectorBase> eigvals_arnoldi(
    const MatrixBase& a,
    int k,
    int m,
    double tol,
    const std::string& result_file
) {
    if (k <= 0 || m <= 0) {
        throw std::invalid_argument("eigvals_arnoldi: k and m must be positive");
    }
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("eigvals_arnoldi: matrix must be square");
    }

    // Symmetric input -> real eigenvalues (Lanczos path, float64 output); otherwise
    // complex128. Symmetry is decided exactly (structure, then an element check).
    const bool symmetric = is_exactly_symmetric(a);
    auto out = ObjectFactory::create_vector(static_cast<uint64_t>(k),
                                            symmetric ? DataType::FLOAT64 : DataType::COMPLEX_FLOAT64,
                                            Structure::VECTOR, result_file);
    ComputeContext::instance().get_device()->eigvals_arnoldi(a, *out, k, m, tol);
    return out;
}

std::unique_ptr<VectorBase> eigvals_skew(
    const MatrixBase& a,
    int k,
    const std::string& result_file
) {
    if (k <= 0) {
        throw std::invalid_argument("eigvals_skew: k must be positive");
    }
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("eigvals_skew: matrix must be square");
    }

    const uint64_t n = a.rows();
    const int kk = std::min(k, static_cast<int>(n));
    auto out = ObjectFactory::create_vector(static_cast<uint64_t>(kk), DataType::COMPLEX_FLOAT64, Structure::VECTOR, result_file);
    ComputeContext::instance().get_device()->eigvals_skew(a, *out, kk);
    return out;
}

std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eig_skew(
    const MatrixBase& a,
    int k,
    const std::string& result_file
) {
    if (k <= 0) {
        throw std::invalid_argument("eig_skew: k must be positive");
    }
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("eig_skew: matrix must be square");
    }

    const uint64_t n = a.rows();
    const int kk = std::min(k, static_cast<int>(n));
    auto w = ObjectFactory::create_vector(static_cast<uint64_t>(kk), DataType::COMPLEX_FLOAT64, Structure::VECTOR, result_file.empty() ? "" : result_file + "_vals");
    auto v = ObjectFactory::create_matrix(n, static_cast<uint64_t>(kk), DataType::COMPLEX_FLOAT64, Structure::DENSE, result_file.empty() ? "" : result_file + "_vecs");

    ComputeContext::instance().get_device()->eig_skew(a, *w, *v, kk);
    return std::make_pair(std::move(w), std::move(v));
}

// Eigen-results follow the input's precision (NumPy; under-promotion, F-047): computed in
// double, returned as float32 / complex64 for float32 / complex64 input.
static bool is_single_precision(const MatrixBase& in) {
    return in.get_data_type() == DataType::FLOAT32 || in.get_data_type() == DataType::COMPLEX_FLOAT32;
}

// The CPU eigensolvers take double precision; other inputs (float32, float16, integers, bits,
// complex64) are converted first. Returns nullptr when `in` is already double precision.
static std::unique_ptr<MatrixBase> as_double_precision(const MatrixBase& in) {
    const DataType dt = in.get_data_type();
    if (dt == DataType::FLOAT64 || dt == DataType::COMPLEX_FLOAT64) return nullptr;
    const bool complex = dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32;
    return cast_to(in, complex ? DataType::COMPLEX_FLOAT64 : DataType::FLOAT64);
}

static std::unique_ptr<VectorBase> narrow_vector(std::unique_ptr<VectorBase> v, bool single) {
    if (!single) return v;
    const uint64_t n = v->size();
    const DataType dt = v->get_data_type();
    if (dt == DataType::FLOAT64) {
        auto out = std::make_unique<DenseVector<float>>(n, "");
        for (uint64_t i = 0; i < n; ++i) out->set(i, static_cast<float>(v->get_element_as_double(i)));
        return out;
    }
    if (dt == DataType::COMPLEX_FLOAT64) {
        auto out = std::make_unique<DenseVector<std::complex<float>>>(n, "");
        for (uint64_t i = 0; i < n; ++i) out->set(i, static_cast<std::complex<float>>(v->get_element_as_complex(i)));
        return out;
    }
    return v;
}

static std::unique_ptr<MatrixBase> narrow_matrix(std::unique_ptr<MatrixBase> m, bool single) {
    if (!single) return m;
    const DataType dt = m->get_data_type();
    if (dt == DataType::FLOAT64) return cast_to(*m, DataType::FLOAT32);
    if (dt == DataType::COMPLEX_FLOAT64) return cast_to(*m, DataType::COMPLEX_FLOAT32);
    return m;
}

std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eig(const MatrixBase& in, const std::string& result_file) {
    if (!is_plain(in)) return eig(*write_out(in), result_file);  // kernels read storage (F-062)
    if (auto d = as_double_precision(in)) {
        auto r = eig(*d, result_file);
        return {narrow_vector(std::move(r.first), is_single_precision(in)), narrow_matrix(std::move(r.second), is_single_precision(in))};
    }
    if (in.rows() != in.cols()) {
        throw std::invalid_argument("eig: matrix must be square");
    }
    uint64_t n = in.rows();
    auto w = ObjectFactory::create_vector(n, DataType::COMPLEX_FLOAT64, Structure::VECTOR, result_file.empty() ? "" : result_file + "_vals");
    auto v = ObjectFactory::create_matrix(n, n, DataType::COMPLEX_FLOAT64, Structure::DENSE, result_file.empty() ? "" : result_file + "_vecs");
    
    ComputeContext::instance().get_device()->eig(in, *w, *v);
    return std::make_pair(narrow_vector(std::move(w), is_single_precision(in)), narrow_matrix(std::move(v), is_single_precision(in)));
}

std::unique_ptr<VectorBase> eigvals(const MatrixBase& in, const std::string& result_file) {
    if (!is_plain(in)) return eigvals(*write_out(in), result_file);  // kernels read storage (F-062)
    if (auto d = as_double_precision(in)) return narrow_vector(eigvals(*d, result_file), is_single_precision(in));
    if (in.rows() != in.cols()) {
        throw std::invalid_argument("eigvals: matrix must be square");
    }
    uint64_t n = in.rows();
    auto w = ObjectFactory::create_vector(n, DataType::COMPLEX_FLOAT64, Structure::VECTOR, result_file);
    ComputeContext::instance().get_device()->eigvals(in, *w);
    return narrow_vector(std::move(w), is_single_precision(in));
}

std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eigh(const MatrixBase& in, const std::string& result_file) {
    if (!is_plain(in)) return eigh(*write_out(in), result_file);  // kernels read storage (F-062)
    if (auto d = as_double_precision(in)) {
        auto r = eigh(*d, result_file);
        return {narrow_vector(std::move(r.first), is_single_precision(in)), narrow_matrix(std::move(r.second), is_single_precision(in))};
    }
    if (in.rows() != in.cols()) {
        throw std::invalid_argument("eigh: matrix must be square");
    }
    uint64_t n = in.rows();
    // eigh returns real eigenvalues for Hermitian matrices
    auto w = ObjectFactory::create_vector(n, DataType::FLOAT64, Structure::VECTOR, result_file.empty() ? "" : result_file + "_vals");
    // Check if input is complex to determine eigenvector type
    // If input is real/symmetric, eigenvectors are real. If complex/hermitian, complex.
    DataType v_type = (in.get_data_type() == DataType::COMPLEX_FLOAT32 || in.get_data_type() == DataType::COMPLEX_FLOAT64) 
                      ? DataType::COMPLEX_FLOAT64 : DataType::FLOAT64;

    auto v = ObjectFactory::create_matrix(n, n, v_type, Structure::DENSE, result_file.empty() ? "" : result_file + "_vecs");
    
    ComputeContext::instance().get_device()->eigh(in, *w, *v, 'L');
    return std::make_pair(narrow_vector(std::move(w), is_single_precision(in)), narrow_matrix(std::move(v), is_single_precision(in)));
}

std::unique_ptr<VectorBase> eigvalsh(const MatrixBase& in, const std::string& result_file) {
    if (!is_plain(in)) return eigvalsh(*write_out(in), result_file);  // kernels read storage (F-062)
    if (auto d = as_double_precision(in)) return narrow_vector(eigvalsh(*d, result_file), is_single_precision(in));
    if (in.rows() != in.cols()) {
        throw std::invalid_argument("eigvalsh: matrix must be square");
    }
    uint64_t n = in.rows();
    auto w = ObjectFactory::create_vector(n, DataType::FLOAT64, Structure::VECTOR, result_file);
    ComputeContext::instance().get_device()->eigvalsh(in, *w, 'L');
    return narrow_vector(std::move(w), is_single_precision(in));
}

}

// --- Special Solvers (Implementation from MatrixOperations.cpp) ---

std::unique_ptr<pycauset::TriangularMatrix<double>> compute_k_matrix(
    const pycauset::TriangularMatrix<bool>& C, 
    double a, 
    const std::string& output_path, 
    int num_threads
) {
    return pycauset::ComputeContext::instance().get_device()->compute_k_matrix(C, a, output_path, num_threads);
}
