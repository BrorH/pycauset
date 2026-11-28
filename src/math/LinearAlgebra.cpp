#include "pycauset/math/DerivedCache.hpp"
#include <cstdio>
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/math/Bitwise.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include <utility>
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/BitClosureMatrix.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/core/Errors.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/StructureResolver.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/math/VectorOps.hpp"
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <functional>
#include <iostream>
#include <complex>
#include <limits>
#include <cstring>
#include <atomic>
#include <optional>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace pycauset {

// --- Helpers ---

namespace {

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

// Structure rule: identity (+ - *) identity of one dtype is a scaled identity, O(1), for every
// dtype whose result dtype is the operands' (bool + bool counts in int32: the general path).
// The factor is the combined value of the two views: conj?(s_a) op conj?(s_b).
static std::unique_ptr<MatrixBase> identity_pair(const MatrixBase& a, const MatrixBase& b, promotion::BinaryOp op,
                                          const char* name, const std::string& result_file) {
    if (!is_identity(a) || !is_identity(b)) return nullptr;
    // Unequal shapes broadcast (a 1 x 1 identity) or raise: the general path decides (it threw
    // where NumPy broadcasts, review 3.3 C-1).
    if (a.rows() != b.rows() || a.cols() != b.cols()) return nullptr;
    (void)name;
    const DataType dt = a.get_data_type();
    if (!result_file.empty() || b.get_data_type() != dt || a.is_subview() || b.is_subview()) return nullptr;
    if (promotion::resolve(op, dt, dt).result_dtype != dt) return nullptr;
    const auto value = [](const MatrixBase& m) {
        return m.is_conjugated() ? std::conj(m.get_scalar()) : m.get_scalar();
    };
    std::complex<double> f;
    switch (op) {
        case promotion::BinaryOp::Add: f = value(a) + value(b); break;
        case promotion::BinaryOp::Subtract: f = value(a) - value(b); break;
        default: f = value(a) * value(b); break;
    }
    auto out = std::unique_ptr<MatrixBase>(static_cast<MatrixBase*>(a.clone().release()));
    out->set_conjugated(false);
    out->set_scalar(f);
    return out;
}

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
    // The closure reads the bits of a strictly upper C (C ± C^T); a diagonal goes the dense way.
    if (static_cast<const TriangularMatrixBase&>(a).has_diagonal()) return nullptr;
    if (promotion::resolve(sign_op > 0 ? promotion::BinaryOp::Add : promotion::BinaryOp::Subtract,
                           DataType::BIT, DataType::BIT).result_dtype != DataType::INT32) return nullptr;
    auto out = std::make_unique<BitClosureMatrix>(a.rows(), a.shared_mapper(), sign_op);
    if (sign_op < 0 && a.is_transposed()) out->set_scalar(-1.0);  // C^T - C = -(C - C^T)
    return out;
}

// Normaliser rule: A^T op B^T for dense A, B of one shape is (A op B)^T. The kernel then reads
// both operands contiguously and the result is returned viewed transposed, as NumPy keeps
// column order (strided reads made dense.T + dense.T 0.27x NumPy). Bit and complex32 storage
// are their own classes and go the general way.
template <typename Op>
static std::unique_ptr<MatrixBase> transposed_pair(const MatrixBase& a, const MatrixBase& b,
                                                   const std::string& result_file, Op op) {
    if (!result_file.empty()) return nullptr;
    if (a.get_structure() != Structure::DENSE || b.get_structure() != Structure::DENSE) return nullptr;
    if (!a.is_transposed() || !b.is_transposed() || a.is_subview() || b.is_subview()) return nullptr;
    if (a.rows() != b.rows() || a.cols() != b.cols()) return nullptr;
    for (DataType dt : {a.get_data_type(), b.get_data_type()}) {
        if (dt == DataType::BIT || dt == DataType::COMPLEX_FLOAT16) return nullptr;
    }
    const auto at = a.transpose();
    const auto bt = b.transpose();
    return op(*at, *bt)->transpose();
}

std::unique_ptr<MatrixBase> add(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (auto r = transposed_pair(a, b, result_file, [](const MatrixBase& x, const MatrixBase& y) { return add(x, y); })) return r;
    if (auto r = identity_pair(a, b, promotion::BinaryOp::Add, "add", result_file)) return r;

    if (auto closure = try_bit_closure(a, b, +1, result_file)) return closure;
    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "add");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();

    DataType res_dtype = promotion::resolve(promotion::BinaryOp::Add, type_a, type_b).result_dtype;
    const auto res_shape = structure::resolve_result_structure(promotion::BinaryOp::Add, a, b, res_dtype);
    const Structure res_mtype = res_shape.structure;
    // Structure rule: bool + bool of one shape counts from the packed words.
    if (type_a == DataType::BIT && type_b == DataType::BIT && res_dtype == DataType::INT32 &&
        res_mtype == Structure::DENSE && result_file.empty() && a.rows() == b.rows() && a.cols() == b.cols()) {
        return bit_counts(a, b, 1);
    }

    std::unique_ptr<MatrixBase> result;
    {
        // A dense result is written completely by the kernel: no zeroing (F-096).
        std::optional<UninitializedStorage> u;
        if (res_mtype == Structure::DENSE && res_dtype != DataType::BIT && result_file.empty()) u.emplace();
        result = ObjectFactory::create_matrix(out_rows, out_cols, res_dtype, res_mtype, result_file,
                                              MatrixOptions{res_shape.has_diagonal});
    }

    ComputeContext::instance().get_device()->add(a, b, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> subtract(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (auto r = transposed_pair(a, b, result_file, [](const MatrixBase& x, const MatrixBase& y) { return subtract(x, y); })) return r;
    if (auto r = identity_pair(a, b, promotion::BinaryOp::Subtract, "subtract", result_file)) return r;

    if (auto closure = try_bit_closure(a, b, -1, result_file)) return closure;
    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "subtract");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();

    DataType res_dtype = promotion::resolve(promotion::BinaryOp::Subtract, type_a, type_b).result_dtype;
    const auto res_shape = structure::resolve_result_structure(promotion::BinaryOp::Subtract, a, b, res_dtype);
    const Structure res_mtype = res_shape.structure;
    // Structure rule: bool - bool of one shape counts from the packed words.
    if (type_a == DataType::BIT && type_b == DataType::BIT && res_dtype == DataType::INT32 &&
        res_mtype == Structure::DENSE && result_file.empty() && a.rows() == b.rows() && a.cols() == b.cols()) {
        return bit_counts(a, b, -1);
    }

    std::unique_ptr<MatrixBase> result;
    {
        // A dense result is written completely by the kernel: no zeroing (F-096).
        std::optional<UninitializedStorage> u;
        if (res_mtype == Structure::DENSE && res_dtype != DataType::BIT && result_file.empty()) u.emplace();
        result = ObjectFactory::create_matrix(out_rows, out_cols, res_dtype, res_mtype, result_file,
                                              MatrixOptions{res_shape.has_diagonal});
    }

    ComputeContext::instance().get_device()->subtract(a, b, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> elementwise_multiply(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (auto r = transposed_pair(a, b, result_file, [](const MatrixBase& x, const MatrixBase& y) { return elementwise_multiply(x, y); })) return r;
    if (auto r = identity_pair(a, b, promotion::BinaryOp::ElementwiseMultiply, "elementwise_multiply", result_file)) return r;

    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "elementwise_multiply");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();

    DataType res_dtype = promotion::resolve(promotion::BinaryOp::ElementwiseMultiply, type_a, type_b).result_dtype;
    const auto res_shape = structure::resolve_result_structure(promotion::BinaryOp::ElementwiseMultiply, a, b, res_dtype);
    const Structure res_mtype = res_shape.structure;
    // Structure rule: bool * bool of one shape is a word-level AND.
    if (type_a == DataType::BIT && type_b == DataType::BIT && res_dtype == DataType::BIT &&
        res_mtype == Structure::DENSE && result_file.empty() && a.rows() == b.rows() && a.cols() == b.cols()) {
        return bitwise(a, b, BitwiseOp::And);
    }

    std::unique_ptr<MatrixBase> result;
    {
        // A dense result is written completely by the kernel: no zeroing (F-096).
        std::optional<UninitializedStorage> u;
        if (res_mtype == Structure::DENSE && res_dtype != DataType::BIT && result_file.empty()) u.emplace();
        result = ObjectFactory::create_matrix(out_rows, out_cols, res_dtype, res_mtype, result_file,
                                              MatrixOptions{res_shape.has_diagonal});
    }

    ComputeContext::instance().get_device()->elementwise_multiply(a, b, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> elementwise_divide(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (auto r = transposed_pair(a, b, result_file, [](const MatrixBase& x, const MatrixBase& y) { return elementwise_divide(x, y); })) return r;
    const auto [out_rows, out_cols] = broadcast_shape_2d(a, b, "elementwise_divide");

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    DataType res_dtype = promotion::resolve(promotion::BinaryOp::Divide, type_a, type_b).result_dtype;
    // Division is not generally structure-preserving (e.g., implicit 0/0 off-triangle).
    // Materialize a dense result for correctness.
    Structure res_mtype = Structure::DENSE;

    std::unique_ptr<MatrixBase> result;
    {
        // A dense result is written completely by the kernel: no zeroing (F-096).
        std::optional<UninitializedStorage> u;
        if (res_mtype == Structure::DENSE && res_dtype != DataType::BIT && result_file.empty()) u.emplace();
        result = ObjectFactory::create_matrix(out_rows, out_cols, res_dtype, res_mtype, result_file);
    }

    ComputeContext::instance().get_device()->elementwise_divide(a, b, *result);

    return result;
}

// --- Vector Operations ---



// a . b in float64 for two real vectors whose dot is float64 (dot() below): the dense kernel
// (fixed blocks, thread-count independent), the unit-vector rule, the typed reader.
static double dot_real(const VectorBase& a, const VectorBase& b) {
    const auto dt_a = a.get_data_type();
    const auto dt_b = b.get_data_type();
    auto plain = [](const VectorBase& v) {
        return v.get_structure() == Structure::VECTOR && v.get_scalar() == std::complex<double>(1.0, 0.0);
    };
    if (plain(a) && plain(b) && dt_a == dt_b && (dt_a == DataType::FLOAT64 || dt_a == DataType::FLOAT32)) {
        return ComputeContext::instance().get_device()->dot(a, b);
    }
    // A unit vector (finite factor): sum_i u_i y_i is u_k y_k plus terms 0 * y_i, which are 0
    // unless y_i is inf or NaN (then NaN, as NumPy computes them). Same value as the full loop
    // below (it accumulates from +0.0, so a -0.0 product gives +0.0 there too); O(n) reads of y
    // instead of per-element virtual reads of both (dot with a unit vector was 0.05x NumPy).
    const auto* ua = dynamic_cast<const UnitVector*>(&a);
    const auto* ub = dynamic_cast<const UnitVector*>(&b);
    if (ua && ub) {  // two unit vectors: one product or none (exact)
        const std::complex<double> x = ua->active_value(), y = ub->active_value();
        if (std::isfinite(x.real()) && std::isfinite(y.real())) {
            debug_trace::set_last("cpu.dot.unit");
            return ua->get_active_index() == ub->get_active_index() ? 0.0 + x.real() * y.real() : 0.0;
        }
    }
    if (const UnitVector* u = ua ? ua : ub) {
        const VectorBase& o = ua ? b : a;
        const std::complex<double> uk = u->active_value();
        const std::complex<double> so = o.get_scalar();
        auto in_place = [&](auto tag) -> std::optional<double> {
            using T = typename decltype(tag)::type;
            const auto* v = dynamic_cast<const DenseVector<T>*>(&o);
            if (!v || so.imag() != 0.0 || o.is_conjugated()) return std::nullopt;
            const T* __restrict p = v->data();
            const double f = so.real();
            const uint64_t k = u->get_active_index(), n = o.size();
            const double yk = static_cast<double>(p[k]) * f;
            // Are all y_i, i != k, finite? One vectorizable pass; a second only when some value
            // is not finite and it may be y_k itself.
            // y * 0 is 0 for finite y and NaN otherwise: a sum of those is NaN iff some y is not
            // finite, and the sum vectorizes (an isfinite test per element did not).
            std::atomic<bool> all{true};
            ParallelForRange(0, n, [&](size_t i0, size_t i1) {
                const T* __restrict q = p;
                if (f == 1.0) {
                    // Unscaled: y is not finite iff its exponent bits are all set (integer
                    // compares vectorize; the floating-point sum below does not).
                    // The exponent is in the high 32-bit word (little-endian); 32-bit compares
                    // vectorize on every x86-64, 64-bit ones need SSE4.1.
                    static_assert(std::endian::native == std::endian::little);
                    constexpr uint32_t E = sizeof(T) == 8 ? 0x7ff00000u : 0x7f800000u;
                    constexpr size_t W = sizeof(T) / 4, HI = W - 1;
                    const uint32_t* __restrict w = reinterpret_cast<const uint32_t*>(q);
                    uint32_t hit = 0;
                    for (size_t i = i0; i < i1; ++i) hit |= static_cast<uint32_t>((w[i * W + HI] & E) == E);
                    if (hit) all = false;
                    return;
                }
                double z0 = 0, z1 = 0, z2 = 0, z3 = 0;
                size_t i = i0;
                for (; i + 4 <= i1; i += 4) {
                    z0 += static_cast<double>(q[i]) * f * 0.0; z1 += static_cast<double>(q[i + 1]) * f * 0.0;
                    z2 += static_cast<double>(q[i + 2]) * f * 0.0; z3 += static_cast<double>(q[i + 3]) * f * 0.0;
                }
                for (; i < i1; ++i) z0 += static_cast<double>(q[i]) * f * 0.0;
                if (std::isnan((z0 + z1) + (z2 + z3))) all = false;
            });
            bool finite = all;
            if (!all && !std::isfinite(yk)) {
                finite = true;
                for (uint64_t i = 0; i < n; ++i)
                    if (i != k && !std::isfinite(static_cast<double>(p[i]) * f)) { finite = false; break; }
            }
            debug_trace::set_last("cpu.dot.unit");
            return finite ? 0.0 + uk.real() * yk : std::numeric_limits<double>::quiet_NaN();
        };
        if (std::isfinite(uk.real())) {
            if (o.get_data_type() == DataType::FLOAT64)
                if (auto r = in_place(type_tag<double>{})) return *r;
            if (o.get_data_type() == DataType::FLOAT32)
                if (auto r = in_place(type_tag<float>{})) return *r;
        }
        if (std::isfinite(uk.real()) && dynamic_cast<const UnitVector*>(&o) == nullptr) {
            std::vector<double> y(o.size());
            if (read_rows(*as_row_matrix(o), 0, 1, y.data())) {
                const uint64_t k = u->get_active_index();
                bool finite = true;
                for (uint64_t i = 0; i < y.size(); ++i) finite &= i == k || std::isfinite(y[i]);
                debug_trace::set_last("cpu.dot.unit");
                return finite ? 0.0 + uk.real() * y[k] : std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    // Every other dtype and view: the typed row reader over the 1 x n bridge.
    std::vector<double> x(a.size()), y(b.size());
    if (read_rows(*as_row_matrix(a), 0, 1, x.data()) && read_rows(*as_row_matrix(b), 0, 1, y.data())) {
        debug_trace::set_last("cpu.dot.typed");
        double acc = 0.0;
        for (uint64_t i = 0; i < x.size(); ++i) acc += x[i] * y[i];
        return acc;
    }
    throw std::logic_error("dot: the typed reader could not read a real vector");
}

std::unique_ptr<MatrixBase> dot(const VectorBase& a, const VectorBase& b) {
    if (a.size() != b.size()) throw std::invalid_argument("dot: vector sizes differ");
    // One rule with matmul (review 3.6 C-9: three implementations; the integer one in the
    // binding returned exact Python ints past int64 where matmul raises, and bool . bool was
    // computed apart): the result dtype is matmul's, and every dtype but float64 is the 1-D
    // case of matmul, 1 x n @ n x 1.
    const DataType r = promotion::resolve(promotion::BinaryOp::Matmul, a.get_data_type(), b.get_data_type()).result_dtype;
    if (r == DataType::FLOAT64 && a.get_scalar().imag() == 0.0 && b.get_scalar().imag() == 0.0) {
        auto out = std::make_unique<DenseMatrix<double>>(1, 1, std::string());
        out->set(0, 0, dot_real(a, b));
        return out;
    }
    auto row = as_row_matrix(a);
    auto col = as_row_matrix(b);
    col->set_transposed(true);  // n x 1
    return dispatch_matmul(*row, *col);
}

double norm(const VectorBase& v) {
    return ComputeContext::instance().get_device()->l2_norm(v);
}

double norm(const MatrixBase& m) {
    return ComputeContext::instance().get_device()->frobenius_norm(m);
}



// Cross product of two 3-vectors, NumPy's cross: NumPy's multiply dtype (bool raises, as
// NumPy's subtract does); integers wrap. float16 / complex32: each product is rounded to the
// dtype, then the difference (NumPy computes every step in the dtype).
std::unique_ptr<VectorBase> cross_product(const VectorBase& a, const VectorBase& b, const std::string&) {
    if (a.size() != 3 || b.size() != 3) throw std::invalid_argument("cross: vectors must have 3 elements");
    const DataType r = promotion::resolve(promotion::BinaryOp::ElementwiseMultiply, a.get_data_type(), b.get_data_type()).result_dtype;
    if (r == DataType::BIT) throw dtype_error("cross: not defined for bool vectors (NumPy's subtract rejects bool)");
    const DataType c = r == DataType::FLOAT16 ? DataType::FLOAT32 : r == DataType::COMPLEX_FLOAT16 ? DataType::COMPLEX_FLOAT32 : r;
    auto ra = to_dense(*as_row_matrix(a), c);
    auto rb = to_dense(*as_row_matrix(b), c);
    auto out = ObjectFactory::create_matrix(1, 3, c, Structure::DENSE, "");
    debug_trace::set_last("cpu.cross");
    visit_dtype(c, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
            const T* x = static_cast<const DenseMatrix<T>&>(*ra).data();
            const T* y = static_cast<const DenseMatrix<T>&>(*rb).data();
            T* z = static_cast<DenseMatrix<T>&>(*out).data();
            auto mul = [&](T p, T q) -> T {
                if constexpr (std::is_integral_v<T>) return static_cast<T>(scalar::WrapMultiplies{}(p, q));
                else if (r == DataType::FLOAT16) return static_cast<T>(static_cast<float>(float16_t(p * q)));
                else return p * q;
            };
            auto sub = [](T p, T q) -> T {
                if constexpr (std::is_integral_v<T>) return static_cast<T>(scalar::WrapMinus{}(p, q));
                else return p - q;
            };
            for (int k = 0; k < 3; ++k) {
                const int i1 = (k + 1) % 3, i2 = (k + 2) % 3;
                z[k] = sub(mul(x[i1], y[i2]), mul(x[i2], y[i1]));
            }
        } else if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            const T* x = static_cast<const DenseMatrix<T>&>(*ra).data();
            const T* y = static_cast<const DenseMatrix<T>&>(*rb).data();
            T* z = static_cast<DenseMatrix<T>&>(*out).data();
            auto mul = [&](T p, T q) -> T {
                if (r != DataType::COMPLEX_FLOAT16) return p * q;
                const T v = p * q;
                return T(static_cast<float>(float16_t(v.real())), static_cast<float>(float16_t(v.imag())));
            };
            for (int k = 0; k < 3; ++k) {
                const int i1 = (k + 1) % 3, i2 = (k + 2) % 3;
                z[k] = mul(x[i1], y[i2]) - mul(x[i2], y[i1]);
            }
        } else {
            throw std::logic_error(std::string("cross: no kernel for ") + dtype_name(c));
        }
    });
    std::unique_ptr<MatrixBase> res = c != r ? cast_to(*out, r) : std::move(out);
    return as_vector(std::move(res), false);
}

// --- Matrix-Vector Operations ---

// Outer product, NumPy's outer: out[i, j] = a[i] * b[j] for vectors of any lengths n and m, in
// NumPy's multiply dtype (integers wrap; bool is logical and). Normaliser: each vector as a
// dense 1 x len row in the computation dtype (as_row_matrix + to_dense: every dtype and view).
// Kernel: one loop per element type; bool copies b's packed row into each row where a[i] is
// set. float16 / complex32 are computed in float32 / complex64 (a product of two float16 is
// exact there) and rounded once. Replaces a square-only, per-dtype, virtual-call kernel that
// read integers through double (CpuSolver::outer_product).
std::unique_ptr<MatrixBase> outer_product(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    const uint64_t n = a.size(), m = b.size();
    const DataType r = promotion::resolve(promotion::BinaryOp::ElementwiseMultiply, a.get_data_type(), b.get_data_type()).result_dtype;
    const DataType c = r == DataType::FLOAT16 ? DataType::FLOAT32 : r == DataType::COMPLEX_FLOAT16 ? DataType::COMPLEX_FLOAT32 : r;
    // bool result: both operands are bool; write_out keeps packed bits (to_dense has no bit target).
    auto row = [c](const VectorBase& v) { return c == DataType::BIT ? write_out(*as_row_matrix(v)) : to_dense(*as_row_matrix(v), c); };
    auto ra = row(a);
    auto rb = row(b);
    auto out = ObjectFactory::create_matrix(n, m, c, Structure::DENSE, c == r ? result_file : std::string());
    debug_trace::set_last("cpu.outer");
    if (c == DataType::BIT) {
        auto& o = static_cast<DenseMatrix<bool>&>(*out);
        const auto& x = static_cast<const DenseMatrix<bool>&>(*ra);
        const auto& y = static_cast<const DenseMatrix<bool>&>(*rb);
        const uint64_t words = o.stride_bytes() / 8;  // same column count: same row stride as y
        uint64_t* dst = o.data();
        const uint64_t* src = y.data();
        ParallelFor(0, n, [&](size_t i) {
            if (x.get(0, i)) std::copy(src, src + words, dst + i * words);
        });
        return out;
    }
    visit_dtype(c, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            if constexpr (!std::is_same_v<T, bool>) {
                const T* x = static_cast<const DenseMatrix<T>&>(*ra).data();
                const T* y = static_cast<const DenseMatrix<T>&>(*rb).data();
                T* z = static_cast<DenseMatrix<T>&>(*out).data();
                ParallelFor(0, n, [&](size_t i) {
                    for (uint64_t j = 0; j < m; ++j) {
                        if constexpr (std::is_integral_v<T>) z[i * m + j] = static_cast<T>(scalar::WrapMultiplies{}(x[i], y[j]));
                        else z[i * m + j] = x[i] * y[j];
                    }
                });
                return;
            }
        }
        throw std::logic_error(std::string("outer: no kernel for ") + dtype_name(c));
    });
    if (c != r) return cast_to(*out, r);
    return out;
}

// Matrix-vector products, NumPy's matmul with a 1-D operand: one path with matmul. The vector
// is an n x 1 column (M @ v) or a 1 x n row (v @ M) over its own storage (as_row_matrix, its
// view kept), so every structure, dtype, view and matmul rule (identity, bit-packed,
// triangular, diagonal, dense; integer overflow raises, F-055) applies. Structure rule: a lazy
// closure C +- C^T over causal bits keeps its two-pass bit kernel (no materialisation) when
// the result is float64. Replaces ~900 lines of hand-written per-dtype matvec / vecmat kernels
// whose structured and complex paths read every element through the virtual accessor.
std::unique_ptr<VectorBase> matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, const std::string&) {
    if (m.cols() != v.size()) throw std::invalid_argument("Matrix columns must match vector size");
    if (auto* cl = dynamic_cast<const BitClosureMatrix*>(&m); cl && !cl->is_materialized()) {
        const DataType vt = v.get_data_type();
        const std::complex<double> k = m.get_scalar();
        const bool real_v = vt != DataType::COMPLEX_FLOAT16 && vt != DataType::COMPLEX_FLOAT32 && vt != DataType::COMPLEX_FLOAT64;
        if (real_v && k.imag() == 0.0 && std::abs(k.real()) < 1073741824.0 &&
            promotion::resolve(promotion::BinaryOp::MatrixVectorMultiply, m.get_data_type(), vt).result_dtype == DataType::FLOAT64) {
            auto out = std::make_unique<DenseVector<double>>(m.rows(), std::string());
            ComputeContext::instance().get_device()->matrix_vector_multiply(m, v, *out);
            return out;
        }
    }
    // Structure rule: a structured matrix (not dense, not identity: that is a scaling) with a
    // float64 / complex128 result is read a block of rows at a time and multiplied by the
    // vector there (CpuSolver), without writing the n x n matrix out.
    {
        const Structure st = m.get_structure();
        const DataType rd = promotion::resolve(promotion::BinaryOp::MatrixVectorMultiply, m.get_data_type(),
                                               v.get_data_type()).result_dtype;
        if (st != Structure::DENSE && st != Structure::IDENTITY && m.rows() > 0 &&
            (rd == DataType::FLOAT64 || rd == DataType::COMPLEX_FLOAT64) &&
            dynamic_cast<const BitClosureMatrix*>(&m) == nullptr) {
            std::unique_ptr<VectorBase> out;
            if (rd == DataType::FLOAT64) out = std::make_unique<DenseVector<double>>(m.rows(), std::string());
            else out = std::make_unique<DenseVector<std::complex<double>>>(m.rows(), std::string());
            ComputeContext::instance().get_device()->matrix_vector_multiply(m, v, *out);
            return out;
        }
    }
    auto col = as_row_matrix(v);
    col->set_transposed(true);  // n x 1
    return as_vector(dispatch_matmul(m, *col), /*transposed=*/false);
}

std::unique_ptr<VectorBase> vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, const std::string&) {
    if (v.size() != m.rows()) throw std::invalid_argument("Vector size must match matrix rows");
    // v @ M = (M^T v)^T: the structured matrix-vector rule above on the transposed view.
    {
        const Structure st = m.get_structure();
        const DataType rd = promotion::resolve(promotion::BinaryOp::MatrixVectorMultiply, m.get_data_type(),
                                               v.get_data_type()).result_dtype;
        if (st != Structure::DENSE && st != Structure::IDENTITY && m.rows() > 0 &&
            (rd == DataType::FLOAT64 || rd == DataType::COMPLEX_FLOAT64) &&
            dynamic_cast<const BitClosureMatrix*>(&m) == nullptr) {
            auto out = matrix_vector_multiply(*m.transpose(), v, "");
            out->set_transposed(v.is_transposed());
            return out;
        }
    }
    auto row = as_row_matrix(v);  // 1 x n
    // NumPy's shapes: (k,) @ (k, n) is 1-D (n,); a transposed (1, k) vector gives the (1, n) row.
    // It was always the row, while ndarray @ M and pc.dot gave (n,) (review 3.6 C-8).
    return as_vector(dispatch_matmul(*row, m), v.is_transposed());
}


namespace {
bool mm_int(DataType d) { return is_integer_or_bit(d) && d != DataType::BIT; }

// The kernels' own scalar handling: the factor of a float operand is folded into the result
// scalar; integer kernels take factor 1 (an integer view factor is written out, so overflow
// and wrapping are decided on the values NumPy sees).
bool mm_factor_ok(const MatrixBase& m) {
    const std::complex<double> s = m.get_scalar();
    const DataType d = m.get_data_type();
    if (is_complex_dtype(d)) return !m.is_conjugated();
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
// A product value as the element type E of a diagonal result (every offered dtype).
template <typename E>
static E diag_elem(std::complex<double> z) {
    if constexpr (std::is_same_v<E, std::complex<float>> || std::is_same_v<E, std::complex<double>> ||
                  std::is_same_v<E, complex_half>) {
        return E(z);
    } else if constexpr (std::is_same_v<E, float16_t>) {
        return float16_t(z.real());
    } else if constexpr (std::is_same_v<E, bool>) {
        return z.real() != 0.0;
    } else {
        return static_cast<E>(z.real());
    }
}

template <typename E>
static E diag_elem_from_int(__int128 v) {
    if constexpr (std::is_integral_v<E> && !std::is_same_v<E, bool>) return static_cast<E>(v);
    else return diag_elem<E>(std::complex<double>(static_cast<double>(v), 0.0));
}

std::unique_ptr<MatrixBase> dispatch_matmul(const MatrixBase& a, const MatrixBase& b, std::string saveas) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Dimension mismatch");
    }
    const DataType da = a.get_data_type(), db = b.get_data_type();
    const DataType r = promotion::resolve(promotion::BinaryOp::Matmul, da, db).result_dtype;
    const Structure sa = a.get_structure(), sb = b.get_structure();
    CpuSolver* device = ComputeContext::instance().get_device();
    const uint64_t m = a.rows(), n = b.cols();

    // R1 identity.
    // I(m, k) @ I(k, n) is I(m, n) only when k >= min(m, n); with a narrower k the product has
    // k ones (IdentityMatrix(3, 2) @ IdentityMatrix(2, 3) is diag(1, 1, 0); review 3.6 C-3).
    if (sa == Structure::IDENTITY && sb == Structure::IDENTITY && a.cols() >= std::min(m, n)) {
        // An identity entry's value is conj?(1 * s) (math/ViewValue.hpp).
        auto res = ObjectFactory::create_matrix(m, n, r, Structure::IDENTITY, saveas);
        res->set_scalar(compose_factors(view_value(1.0, a.get_scalar(), a.is_conjugated()),
                                        view_value(1.0, b.get_scalar(), b.is_conjugated())));
        return res;
    }
    auto identity_times = [&](const MatrixBase& id, const MatrixBase& other) {
        const std::complex<double> s = view_value(1.0, id.get_scalar(), id.is_conjugated());
        // An integer result multiplies by an exact integer: a Python float 1.0 promoted int64 /
        // uint64 to float64 and back, rounding above 2^53 (review 3.3 C-2).
        const bool int_factor = is_integer_or_bit(r) && scalar::is_exact_int_factor(s);
        const ScalarArg arg = int_factor ? ScalarArg::from_int64(static_cast<int64_t>(s.real()))
                            : s.imag() == 0.0 ? ScalarArg::from_double(s.real()) : ScalarArg::from_complex(s);
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
        if (sa == Structure::TRIANGULAR && sb == Structure::TRIANGULAR && !a.is_transposed() && !b.is_transposed() &&
            !mm_has_diagonal(a) && !mm_has_diagonal(b)) {  // the path-count kernel: strict operands
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

    // R4 diagonal: the typed diagonal reader (any dtype); integer products are exact (128 bits)
    // and raise when they leave the result dtype (F-055).
    if (sa == Structure::DIAGONAL && sb == Structure::DIAGONAL && m == n && r != DataType::BIT &&
        !a.is_subview() && !b.is_subview()) {
        auto res = ObjectFactory::create_matrix(m, n, r, Structure::DIAGONAL, saveas);
        auto write = [&](auto&& value_at) {
            visit_matrix(*res, [&](auto& typed) {
                using E = std::decay_t<decltype(typed.get(0, 0))>;
                if constexpr (requires { typed.set_diagonal(uint64_t{0}, E{}); }) {
                    for (uint64_t i = 0; i < m; ++i) typed.set_diagonal(i, value_at.template operator()<E>(i));
                } else {
                    throw std::logic_error("matmul: diagonal result has no typed setter");
                }
            });
        };
        if (mm_int(r)) {
            std::vector<int64_t> x(m), y(m);
            if (read_diagonal(a, x.data()) && read_diagonal(b, y.data())) {
                const bool u64 = r == DataType::UINT64;
                const auto [lo_d, hi_d] = mm_int_range(r);
                // Compared exactly: as doubles, 2**63 (one past INT64_MAX) passed the check and
                // wrapped (review 3.6 C-1). The range ends are exact integers in double, except
                // the 64-bit maxima, which are 2**63 - 1 / 2**64 - 1.
                const __int128 lo = static_cast<__int128>(lo_d);
                const __int128 hi = r == DataType::INT64 ? static_cast<__int128>(std::numeric_limits<int64_t>::max())
                                  : u64 ? static_cast<__int128>(std::numeric_limits<uint64_t>::max())
                                        : static_cast<__int128>(hi_d);
                std::vector<__int128> p(m);
                for (uint64_t i = 0; i < m; ++i) {
                    p[i] = u64 ? static_cast<__int128>(static_cast<uint64_t>(x[i])) * static_cast<uint64_t>(y[i])
                               : static_cast<__int128>(x[i]) * y[i];
                    if (p[i] < lo || p[i] > hi) {
                        throw std::overflow_error("Integer matmul overflow: result does not fit in output dtype");
                    }
                }
                debug_trace::set_last("cpu.matmul.diagonal");
                write([&]<typename E>(uint64_t i) { return diag_elem_from_int<E>(p[i]); });
                return res;
            }
        } else if (is_complex_dtype(r)) {
            std::vector<std::complex<double>> x(m), y(m);
            if (read_diagonal(a, x.data()) && read_diagonal(b, y.data())) {
                debug_trace::set_last("cpu.matmul.diagonal");
                write([&]<typename E>(uint64_t i) { return diag_elem<E>(x[i] * y[i]); });
                return res;
            }
        } else {
            std::vector<double> x(m), y(m);
            if (read_diagonal(a, x.data()) && read_diagonal(b, y.data())) {
                debug_trace::set_last("cpu.matmul.diagonal");
                write([&]<typename E>(uint64_t i) { return diag_elem<E>(std::complex<double>(x[i] * y[i], 0.0)); });
                return res;
            }
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

// Result dtype of solve/svd (LAPACK dtypes), NumPy's rule: integers and bit solve in float64, float16/float32 in
// float32, complex stays complex (complex64 only when nothing needs double precision).
DataType lapack_dtype(DataType a, DataType b) {
    auto single = [](DataType d) {
        return d == DataType::FLOAT16 || d == DataType::FLOAT32 || d == DataType::COMPLEX_FLOAT16 ||
               d == DataType::COMPLEX_FLOAT32;
    };
    const bool sp = single(a) && single(b);
    if (is_complex_dtype(a) || is_complex_dtype(b)) return sp ? DataType::COMPLEX_FLOAT32 : DataType::COMPLEX_FLOAT64;
    return sp ? DataType::FLOAT32 : DataType::FLOAT64;
}

// The LAPACK kernels (compute/cpu/CpuLinalg.cpp) read a plain dense matrix in a LAPACK dtype;
// every other dtype, structure and view is written out with the shared typed reader first.
static std::unique_ptr<MatrixBase> lapack_input(const MatrixBase& a, DataType& dt) {
    dt = lapack_dtype(a.get_data_type(), a.get_data_type());
    if (a.get_structure() == Structure::DENSE && a.get_data_type() == dt && is_plain(a)) return nullptr;
    return to_dense(a, dt);
}

static DataType real_dtype(DataType d) {
    return d == DataType::COMPLEX_FLOAT32 ? DataType::FLOAT32 : d == DataType::COMPLEX_FLOAT64 ? DataType::FLOAT64 : d;
}

static DataType complex_dtype(DataType d) {
    return d == DataType::FLOAT32 ? DataType::COMPLEX_FLOAT32 : d == DataType::FLOAT64 ? DataType::COMPLEX_FLOAT64 : d;
}

static void require_square(const MatrixBase& a, const char* op) {
    if (a.rows() != a.cols()) throw std::invalid_argument(std::string(op) + ": matrix must be square");
}

// Cholesky, NumPy's cholesky: lower L with A = L L^H, reading the lower triangle; any dtype,
// structure and view in the LAPACK dtype. Not positive definite: std::domain_error.
std::unique_ptr<MatrixBase> cholesky(const MatrixBase& a, const std::string& result_file) {
    require_square(a, "cholesky");
    DataType dt;
    auto h = lapack_input(a, dt);
    auto out = ObjectFactory::create_matrix(a.rows(), a.cols(), dt, Structure::DENSE, result_file);
    ComputeContext::instance().get_device()->cholesky(h ? *h : a, *out);
    return out;
}

// LU with partial pivoting, SciPy's lu (NumPy has none): A = P L U with P (m, m) real,
// L (m, k) unit lower, U (k, n); any dtype, structure and view in the LAPACK dtype.
std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>> lu(const MatrixBase& a, const std::string&)
{
    DataType dt;
    auto h = lapack_input(a, dt);
    const uint64_t m = a.rows(), n = a.cols(), k = std::min(m, n);
    auto p = ObjectFactory::create_matrix(m, m, real_dtype(dt), Structure::DENSE, "");
    auto l = ObjectFactory::create_matrix(m, k, dt, Structure::DENSE, "");
    auto u = ObjectFactory::create_matrix(k, n, dt, Structure::DENSE, "");
    ComputeContext::instance().get_device()->lu(h ? *h : a, *p, *l, *u);
    return std::make_tuple(std::move(p), std::move(l), std::move(u));
}

// QR, NumPy's qr: reduced Q (m, k) and R (k, n), or complete Q (m, m) and R (m, n); any dtype,
// structure and view in the LAPACK dtype.
std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>> qr(const MatrixBase& a, bool complete) {
    DataType dt;
    auto h = lapack_input(a, dt);
    const uint64_t m = a.rows(), n = a.cols(), k = complete ? m : std::min(m, n);
    auto q = ObjectFactory::create_matrix(m, k, dt, Structure::DENSE, "");
    auto r = ObjectFactory::create_matrix(k, n, dt, Structure::DENSE, "");
    ComputeContext::instance().get_device()->qr(h ? *h : a, *q, *r);
    return std::make_tuple(std::move(q), std::move(r));
}

std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> svd(
    const MatrixBase& a, bool full_matrices, bool compute_uv)
{
    // The LAPACK kernel reads dense storage in a LAPACK dtype: integers/bit in float64,
    // float16 in float32, complex32 in complex64 (NumPy's rule); any structure or view is
    // written out first (F-062, F-079: complex SVD used to fall back to NumPy).
    {
        const DataType in = a.get_data_type();
        const DataType want = lapack_dtype(in, in);
        if (a.get_structure() != Structure::DENSE || in != want || !is_plain(a)) {
            return svd(*to_dense(a, want), full_matrices, compute_uv);
        }
    }
    const uint64_t m = a.rows(), n = a.cols(), k = std::min(m, n);
    const DataType dtype = a.get_data_type();
    const DataType s_dtype = dtype == DataType::COMPLEX_FLOAT32 ? DataType::FLOAT32
                           : dtype == DataType::COMPLEX_FLOAT64 ? DataType::FLOAT64 : dtype;
    // NumPy shapes: U (m, m) and VT (n, n) when full_matrices, else (m, k) and (k, n).
    // Values only: U m x 0 and VT 0 x n tell the kernel (?gesdd job 'N').
    auto u = ObjectFactory::create_matrix(m, !compute_uv ? 0 : full_matrices ? m : k, dtype, Structure::DENSE, "");
    auto s = ObjectFactory::create_vector(k, s_dtype, Structure::VECTOR, "");
    auto vt = ObjectFactory::create_matrix(!compute_uv ? 0 : full_matrices ? n : k, n, dtype, Structure::DENSE, "");
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
    const DataType dt = lapack_dtype(a.get_data_type(), b.get_data_type());
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

std::unique_ptr<MatrixBase> lstsq(const MatrixBase& a, const MatrixBase& b)
{
    if (a.rows() != b.rows()) {
        throw std::invalid_argument("lstsq: 'b' must have the same number of rows as 'a'");
    }
    // As solve: the kernel reads dense storage of one LAPACK dtype; anything else is written out.
    const DataType dt = lapack_dtype(a.get_data_type(), b.get_data_type());
    auto ready = [dt](const MatrixBase& m) {
        return m.get_structure() == Structure::DENSE && m.get_data_type() == dt && is_plain(m);
    };
    std::unique_ptr<MatrixBase> pa = ready(a) ? nullptr : to_dense(a, dt);
    std::unique_ptr<MatrixBase> pb = ready(b) ? nullptr : to_dense(b, dt);
    const MatrixBase& A = pa ? *pa : a;
    const MatrixBase& B = pb ? *pb : b;
    auto x = ObjectFactory::create_matrix(A.cols(), B.cols(), dt, Structure::DENSE, "");
    ComputeContext::instance().get_device()->lstsq(A, B, *x);
    return x;
}

// Matrix inverse, as NumPy's inv computes it: solve(A, I) by LU with partial pivoting (?gesv)
// in the LAPACK dtype (solve_dtype: integers and bit in float64, float16/32 in float32, complex
// stays complex). Singular input raises. Structure rules (review Adding Operations §3):
//   identity:            (s I)^-1 = I / s, O(1)
//   diagonal:            reciprocal diagonal, O(n)
//   strictly triangular: singular by structure (zero diagonal)
//   upper triangular:    the inverse is upper triangular (partial pivoting swaps no rows of an
//                        upper triangle, so the dense solve's lower part is exactly zero)
// Replaces the per-class inverse() methods (float32/float64 dense only, a triangular one that
// called every triangle singular) and the silent NumPy fallback in Python.
std::unique_ptr<MatrixBase> inverse(const MatrixBase& a) {
    if (a.rows() != a.cols()) throw std::invalid_argument("inverse: matrix must be square");
    const uint64_t n = a.rows();
    const DataType in = a.get_data_type();
    const DataType dt = lapack_dtype(in, in);
    const Structure st = a.get_structure();
    auto singular = []() -> std::unique_ptr<MatrixBase> { throw linalg_error("inverse: matrix is singular"); };
    auto set_complex = [](MatrixBase& m, uint64_t i, uint64_t j, std::complex<double> z) {
        visit_matrix(m, [&](auto& typed) {
            using E = std::decay_t<decltype(typed.get(0, 0))>;
            if constexpr (std::is_same_v<E, float> || std::is_same_v<E, double>) typed.set(i, j, static_cast<E>(z.real()));
            else if constexpr (std::is_same_v<E, std::complex<float>> || std::is_same_v<E, std::complex<double>>)
                typed.set(i, j, E(z));
            else throw std::logic_error("inverse: result is not in a LAPACK dtype");
        });
    };
    if (n == 0) return make_matrix(Structure::DENSE, dt, 0, 0);

    if (st == Structure::IDENTITY) {
        const std::complex<double> s = view_value(1.0, a.get_scalar(), a.is_conjugated());
        if (s == std::complex<double>(0.0, 0.0)) return singular();
        auto out = make_matrix(Structure::IDENTITY, dt, n, n);
        out->set_scalar(1.0 / s);
        return out;
    }
    if (st == Structure::TRIANGULAR && !static_cast<const TriangularMatrixBase&>(a).has_diagonal()) return singular();
    if (st == Structure::DIAGONAL) {
        std::vector<std::complex<double>> d(n);
        if (read_diagonal(a, d.data())) {  // typed (windows of a diagonal read the general path)
            auto out = make_matrix(Structure::DIAGONAL, dt, n, n);
            for (uint64_t i = 0; i < n; ++i) {
                if (d[i] == std::complex<double>(0.0, 0.0)) return singular();
                set_complex(*out, i, i, scalar::true_divide(std::complex<double>(1.0, 0.0), d[i]));
            }
            return out;
        }
    }

    auto eye = make_matrix(Structure::DENSE, dt, n, n);
    for (uint64_t i = 0; i < n; ++i) set_complex(*eye, i, i, 1.0);
    auto x = solve(a, *eye, "");

    if (st == Structure::TRIANGULAR && !a.is_transposed() && !a.is_subview()) {
        return pack_as(*x, Structure::TRIANGULAR, /*has_diagonal=*/true);  // upper part, typed
    }
    return x;
}

// Exact symmetry test for real matrices (no tolerance: a "nearly symmetric" matrix is
// treated as general, which is always correct, just slower).
static bool is_exactly_symmetric(const MatrixBase& a) {
    if (a.rows() != a.cols()) return false;
    const DataType dt = a.get_data_type();
    if (is_complex_dtype(dt)) {
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

// The iterative eigen entries take real input only: their kernels read stored values as
// double (complex input lost its imaginary part or failed inside, review 3.6 C-10).
static void require_real(const MatrixBase& a, const char* op) {
    if (is_complex_dtype(a.get_data_type())) {
        throw dtype_error(std::string(op) + ": real matrices only (bool, integer, float); got " +
                                dtype_name(a.get_data_type()));
    }
}

// Exact skew-symmetry test (A == -A^T, element by element: a guess may not decide which
// algorithm's answer is returned). The skew kernels read one triangle, so a symmetric input
// gave imaginary eigenvalues for a real spectrum.
static void require_skew(const MatrixBase& a, const char* op) {
    const uint64_t n = a.rows();
    for (uint64_t i = 0; i < n; ++i) {
        if (a.get_element_as_double(i, i) != 0.0) {
            throw std::invalid_argument(std::string(op) + ": the matrix is not skew-symmetric (A[" +
                                        std::to_string(i) + ", " + std::to_string(i) + "] != 0)");
        }
        for (uint64_t j = i + 1; j < n; ++j) {
            if (a.get_element_as_double(i, j) != -a.get_element_as_double(j, i)) {
                throw std::invalid_argument(std::string(op) + ": the matrix is not skew-symmetric (A[" +
                                            std::to_string(i) + ", " + std::to_string(j) + "] != -A[" +
                                            std::to_string(j) + ", " + std::to_string(i) + "])");
            }
        }
    }
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
    require_real(a, "eigvals_arnoldi");

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
    require_real(a, "eigvals_skew");
    require_skew(a, "eigvals_skew");

    const uint64_t n = a.rows();
    const int kk = static_cast<int>(std::min<uint64_t>(static_cast<uint64_t>(k), n));
    auto out = ObjectFactory::create_vector(static_cast<uint64_t>(kk), DataType::COMPLEX_FLOAT64, Structure::VECTOR, result_file);
    if (kk > 0) ComputeContext::instance().get_device()->eigvals_skew(a, *out, kk);  // 0 x 0: empty
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
    require_real(a, "eig_skew");
    require_skew(a, "eig_skew");

    const uint64_t n = a.rows();
    const int kk = static_cast<int>(std::min<uint64_t>(static_cast<uint64_t>(k), n));
    auto w = ObjectFactory::create_vector(static_cast<uint64_t>(kk), DataType::COMPLEX_FLOAT64, Structure::VECTOR, result_file.empty() ? "" : result_file + "_vals");
    auto v = ObjectFactory::create_matrix(n, static_cast<uint64_t>(kk), DataType::COMPLEX_FLOAT64, Structure::DENSE, result_file.empty() ? "" : result_file + "_vecs");

    if (kk > 0) ComputeContext::instance().get_device()->eig_skew(a, *w, *v, kk);  // 0 x 0: empty
    return std::make_pair(std::move(w), std::move(v));
}

// |x| of every element of a (any dtype, structure, view) as a dense row-major float64 buffer,
// through the shared typed reader (to_dense in float64 or complex128).
static std::vector<double> abs_values(const MatrixBase& a) {
    const DataType dt = a.get_data_type();
    const bool cplx = is_complex_dtype(dt);
    auto d = to_dense(a, cplx ? DataType::COMPLEX_FLOAT64 : DataType::FLOAT64);
    const uint64_t total = a.rows() * a.cols();
    std::vector<double> out(total);
    if (cplx) {
        const auto* p = static_cast<const DenseMatrix<std::complex<double>>&>(*d).data();
        for (uint64_t i = 0; i < total; ++i) out[i] = std::abs(p[i]);
    } else {
        const auto* p = static_cast<const DenseMatrix<double>&>(*d).data();
        for (uint64_t i = 0; i < total; ++i) out[i] = std::abs(p[i]);
    }
    return out;
}

static std::vector<double> singular_values(const MatrixBase& a) {
    auto s = std::get<1>(svd(a, false, false));
    std::vector<double> out(s->size());
    for (uint64_t i = 0; i < s->size(); ++i) out[i] = s->get_element_as_double(i);
    return out;
}

// NumPy's matrix norms: 'fro', 'nuc', 2 / -2 (largest / smallest singular value), 1 / -1
// (max / min column sum of |a|), inf / -inf (max / min row sum). Computed in double precision.
static double matrix_norm_compute(const MatrixBase& a, const std::string& ord) {
    if (ord == "fro") return norm(a);
    if (ord == "nuc" || ord == "2" || ord == "-2") {
        const auto s = singular_values(a);
        if (s.empty()) throw std::invalid_argument("norm: empty matrix");
        if (ord == "nuc") return std::accumulate(s.begin(), s.end(), 0.0);
        return ord == "2" ? *std::max_element(s.begin(), s.end()) : *std::min_element(s.begin(), s.end());
    }
    const bool cols = ord == "1" || ord == "-1";
    if (!cols && ord != "inf" && ord != "-inf") throw std::invalid_argument("norm: invalid norm order '" + ord + "' for matrices");
    const uint64_t m = a.rows(), n = a.cols();
    if (m == 0 || n == 0) throw std::invalid_argument("norm: empty matrix");
    const auto v = abs_values(a);
    std::vector<double> sums(cols ? n : m, 0.0);
    for (uint64_t i = 0; i < m; ++i)
        for (uint64_t j = 0; j < n; ++j) sums[cols ? j : i] += v[i * n + j];
    const bool largest = ord == "1" || ord == "inf";
    return largest ? *std::max_element(sums.begin(), sums.end()) : *std::min_element(sums.begin(), sums.end());
}

double matrix_norm(const MatrixBase& a, const std::string& ord) {
    return cached<double>(a, "norm", ord, [&] { return matrix_norm_compute(a, ord); });
}

// NumPy's vector norms: p = inf (max |x|), -inf (min |x|), 0 (count of non-zeros), otherwise
// (sum |x|^p)^(1/p). Computed in double precision.
double vector_norm(const VectorBase& v, double p) {
    const auto x = abs_values(*as_row_matrix(v));
    if (std::isinf(p)) {
        if (x.empty()) throw std::invalid_argument("norm: empty vector");
        return p > 0 ? *std::max_element(x.begin(), x.end()) : *std::min_element(x.begin(), x.end());
    }
    if (p == 0) return static_cast<double>(std::count_if(x.begin(), x.end(), [](double t) { return t != 0.0; }));
    if (p == 2) return norm(v);
    double acc = 0;
    for (double t : x) acc += p == 1 ? t : std::pow(t, p);
    return p == 1 ? acc : std::pow(acc, 1.0 / p);
}

// NumPy's matrix_rank: singular values above tol; tol < 0 means NumPy's default
// max(s) * max(m, n) * eps (eps of the SVD's precision).
static int64_t matrix_rank_compute(const MatrixBase& a, double tol) {
    if (a.rows() == 0 || a.cols() == 0) return 0;
    const auto s = singular_values(a);
    if (tol < 0) {
        const bool single = lapack_dtype(a.get_data_type(), a.get_data_type()) == DataType::FLOAT32 ||
                            lapack_dtype(a.get_data_type(), a.get_data_type()) == DataType::COMPLEX_FLOAT32;
        const double eps = single ? std::numeric_limits<float>::epsilon() : std::numeric_limits<double>::epsilon();
        tol = *std::max_element(s.begin(), s.end()) * static_cast<double>(std::max(a.rows(), a.cols())) * eps;
    }
    return std::count_if(s.begin(), s.end(), [tol](double t) { return t > tol; });
}

int64_t matrix_rank(const MatrixBase& a, double tol) {
    char p[32];
    std::snprintf(p, sizeof p, "%a", tol);
    return cached<int64_t>(a, "rank", p, [&] { return matrix_rank_compute(a, tol); });
}

// Eigen decompositions, NumPy's eig / eigvals / eigh / eigvalsh: any dtype, structure and view
// in the LAPACK dtype (float16 and float32 in single precision, as NumPy). eig: complex
// eigenvalues and right eigenvectors (?geev); eigh: real ascending eigenvalues of the lower
// triangle (?syevd / ?heevd) and eigenvectors in the input's LAPACK dtype.
std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eig(const MatrixBase& in, const std::string&) {
    require_square(in, "eig");
    DataType dt;
    auto h = lapack_input(in, dt);
    const uint64_t n = in.rows();
    auto w = ObjectFactory::create_vector(n, complex_dtype(dt), Structure::VECTOR, "");
    auto v = ObjectFactory::create_matrix(n, n, complex_dtype(dt), Structure::DENSE, "");
    ComputeContext::instance().get_device()->eig(h ? *h : in, *w, *v);
    return std::make_pair(std::move(w), std::move(v));
}

static std::unique_ptr<VectorBase> eigvals_compute(const MatrixBase& in) {
    require_square(in, "eigvals");
    DataType dt;
    auto h = lapack_input(in, dt);
    auto w = ObjectFactory::create_vector(in.rows(), complex_dtype(dt), Structure::VECTOR, "");
    ComputeContext::instance().get_device()->eigvals(h ? *h : in, *w);
    return w;
}

std::unique_ptr<VectorBase> eigvals(const MatrixBase& in, const std::string&) {
    return cached_vector(in, "eigvals", [&] { return eigvals_compute(in); });
}

std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eigh(const MatrixBase& in, const std::string&) {
    require_square(in, "eigh");
    DataType dt;
    auto h = lapack_input(in, dt);
    const uint64_t n = in.rows();
    auto w = ObjectFactory::create_vector(n, real_dtype(dt), Structure::VECTOR, "");
    auto v = ObjectFactory::create_matrix(n, n, dt, Structure::DENSE, "");
    ComputeContext::instance().get_device()->eigh(h ? *h : in, *w, *v, 'L');
    return std::make_pair(std::move(w), std::move(v));
}

static std::unique_ptr<VectorBase> eigvalsh_compute(const MatrixBase& in) {
    require_square(in, "eigvalsh");
    DataType dt;
    auto h = lapack_input(in, dt);
    auto w = ObjectFactory::create_vector(in.rows(), real_dtype(dt), Structure::VECTOR, "");
    ComputeContext::instance().get_device()->eigvalsh(h ? *h : in, *w, 'L');
    return w;
}

std::unique_ptr<VectorBase> eigvalsh(const MatrixBase& in, const std::string&) {
    return cached_vector(in, "eigvalsh", [&] { return eigvalsh_compute(in); });
}

// Moore-Penrose pseudoinverse, NumPy's pinv: A = U S V^H, A+ = V S+ U^H, singular values at or
// below max(m, n) * eps * max(s) treated as zero (NumPy's default rtol). V S+ is formed in
// place (columns of V^H scaled) and multiplied by U^H through dispatch_matmul.
std::unique_ptr<MatrixBase> pinv(const MatrixBase& a) {
    auto [u, s, vt] = svd(a, false);
    const uint64_t m = a.rows(), n = a.cols(), k = s->size();
    const DataType dt = u->get_data_type();
    return visit_dtype(dt, [&](auto tag) -> std::unique_ptr<MatrixBase> {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            using R = std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>, float, double>;
            const R* sp = static_cast<const DenseVector<R>&>(*s).data();
            R smax = 0;
            for (uint64_t l = 0; l < k; ++l) smax = std::max(smax, sp[l]);
            const R cutoff = static_cast<R>(std::max(m, n)) * std::numeric_limits<R>::epsilon() * smax;
            auto conj = [](T x) {
                if constexpr (std::is_same_v<T, R>) return x;
                else return std::conj(x);
            };
            // VS: n x k, VS[i][l] = conj(VT[l][i]) / s_l (0 for s_l <= cutoff). UH: k x m.
            const T* up = static_cast<const DenseMatrix<T>&>(*u).data();
            const T* vp = static_cast<const DenseMatrix<T>&>(*vt).data();
            auto vs = std::make_unique<DenseMatrix<T>>(n, k, std::string());
            auto uh = std::make_unique<DenseMatrix<T>>(k, m, std::string());
            T* vsp = vs->data();
            T* uhp = uh->data();
            for (uint64_t l = 0; l < k; ++l) {
                const R inv = sp[l] > cutoff ? R(1) / sp[l] : R(0);
                for (uint64_t i = 0; i < n; ++i) vsp[i * k + l] = conj(vp[l * n + i]) * inv;
                for (uint64_t j = 0; j < m; ++j) uhp[l * m + j] = conj(up[j * k + l]);
            }
            debug_trace::set_last("cpu.pinv.svd");
            if (k == 0) return make_matrix(Structure::DENSE, dt, n, m);
            return dispatch_matmul(*vs, *uh);
        } else {
            throw std::logic_error("pinv: SVD gave a non-LAPACK dtype");
        }
    });
}

}

// --- Special Solvers (Implementation from MatrixOperations.cpp) ---

std::unique_ptr<pycauset::MatrixBase> compute_k_matrix(const pycauset::MatrixBase& c, double a) {
    // The kernel reads the stored bits by rows: it ignored a transpose (compute_k(C.T) was K of
    // C) and divided by a = 0 (review 3.4 C-4).
    const auto* C = dynamic_cast<const pycauset::TriangularMatrix<bool>*>(&c);
    if (C == nullptr) {
        throw pycauset::dtype_error("compute_k: expects a causal matrix (TriangularMatrix of dtype bool), got " +
                                    std::string(pycauset::dtype_name(c.get_data_type())));
    }
    if (C->has_diagonal()) {
        throw std::invalid_argument("compute_k: expects a causal (strictly upper triangular) matrix");
    }
    if (c.is_subview() || c.get_scalar() != std::complex<double>(1.0, 0.0)) {
        throw std::invalid_argument("compute_k: expects a whole causal matrix (not a window or scaled view)");
    }
    if (a == 0.0) throw pycauset::linalg_error("compute_k: aI + C is singular for a = 0");
    auto& device = *pycauset::ComputeContext::instance().get_device();
    if (c.is_transposed()) {
        const auto base = c.transpose();
        return device.compute_k_matrix(static_cast<const pycauset::TriangularMatrix<bool>&>(*base), a, "")->transpose();
    }
    return device.compute_k_matrix(*C, a, "");
}
