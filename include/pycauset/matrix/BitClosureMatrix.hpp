#pragma once
// C + C^T (symmetric) or C - C^T (antisymmetric) for a causal (bit, strictly upper
// triangular) matrix C, read straight from C's bits: LAZY.md L2b.
//
// The comparability matrix and the Pauli-Jordan-like C - C^T are the central objects of a
// causal set. Written out they take n^2/2 int32 elements (L2a); this view keeps the n^2/2
// *bits* of C (32x less) and computes elements on read. Matrix-vector products use the bits
// directly (two triangular passes), so iterative eigensolvers never materialize anything.
//
// Contract (LAZY.md section 2): indistinguishable from the eager result except for speed and
// memory. Logical dtype int32 (the bit +- bit promotion), structure symmetric/antisymmetric,
// value semantics (shares C's storage copy-on-write; later writes to C don't leak in), and
// writable: the first write (or an operation without a closure-aware path) materializes the
// packed result in place, so the Python object keeps its identity.

#include <memory>
#include <stdexcept>
#include <string>

#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"

namespace pycauset {

class BitClosureMatrix : public AntiSymmetricMatrixBase {
public:
    // sign = +1: C + C^T; sign = -1: C - C^T. `mapper` is C's storage (shared, CoW).
    BitClosureMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper, int sign)
        : AntiSymmetricMatrixBase(n, mapper, sign > 0 ? Structure::SYMMETRIC : Structure::ANTISYMMETRIC,
                                  DataType::INT32),
          bits_(std::make_shared<TriangularMatrix<bool>>(n, mapper)),
          sign_(sign > 0 ? 1 : -1) {}

    int sign() const { return sign_; }
    bool is_materialized() const { return materialized_ != nullptr; }
    const TriangularMatrix<bool>& bits() const { return *bits_; }

    // Element without the view factor.
    int32_t base_get(uint64_t i, uint64_t j) const {
        if (materialized_) return materialized_->get(i, j);
        if (i == j) return 0;
        const int32_t c_ij = (i < j) ? (bits_->get(i, j) ? 1 : 0) : 0;
        const int32_t c_ji = (j < i) ? (bits_->get(j, i) ? 1 : 0) : 0;
        return c_ij + sign_ * c_ji;
    }

    // Typed get follows the codebase convention: the stored value WITHOUT the view factor
    // (callers apply it, see element_get / exact_int_reader). Use value() for the logical one.
    int32_t get(uint64_t i, uint64_t j) const {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        return base_get(i, j);
    }

    // Logical element (view factor applied with NumPy's integer wrapping).
    int32_t value(uint64_t i, uint64_t j) const {
        const int32_t v = get(i, j);
        if (scalar_ == 1.0) return v;
        return scalar::wrap_scale(v, static_cast<int64_t>(scalar_.real()));
    }

    void set(uint64_t i, uint64_t j, int32_t value) {
        materialize_in_place();
        materialized_->set(i, j, value);
    }

    double element_as_double(uint64_t i, uint64_t j) const override { return static_cast<double>(value(i, j)); }
    std::complex<double> element_as_complex(uint64_t i, uint64_t j) const override {
        return {static_cast<double>(value(i, j)), 0.0};
    }
    void set_element_double(uint64_t i, uint64_t j, double value) override {
        set(i, j, static_cast<int32_t>(value));
    }

    // Packed result with the factor folded in (used by save, export, unsupported ops).
    std::unique_ptr<MatrixBase> materialized_copy() const {
        const uint64_t n = rows();
        auto out = std::make_unique<SymmetricMatrix<int32_t>>(n, "", sign_ < 0);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = sign_ < 0 ? i + 1 : i; j < n; ++j) out->set(i, j, value(i, j));
        }
        return out;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(double s, const std::string& f = "") const override {
        return materialized_copy()->multiply_scalar(s, f);
    }
    std::unique_ptr<MatrixBase> multiply_scalar(int64_t s, const std::string& f = "") const override {
        return materialized_copy()->multiply_scalar(s, f);
    }
    std::unique_ptr<MatrixBase> add_scalar(double s, const std::string& f = "") const override {
        return materialized_copy()->add_scalar(s, f);
    }
    std::unique_ptr<MatrixBase> add_scalar(int64_t s, const std::string& f = "") const override {
        return materialized_copy()->add_scalar(s, f);
    }

    // (C + C^T)^T = itself; (C - C^T)^T = -(C - C^T): a factor of -1.
    std::unique_ptr<MatrixBase> transpose(const std::string& = "") const override {
        auto c = clone();
        auto out = std::unique_ptr<MatrixBase>(static_cast<MatrixBase*>(c.release()));
        if (sign_ < 0) out->set_scalar(out->get_scalar() * -1.0);
        return out;
    }

    std::unique_ptr<PersistentObject> clone() const override {
        auto out = std::make_unique<BitClosureMatrix>(rows(), mapper_, sign_);
        if (materialized_) {
            auto m = materialized_->clone();  // shares packed storage (CoW)
            out->materialized_.reset(static_cast<SymmetricMatrix<int32_t>*>(m.release()));
        }
        out->set_scalar(get_scalar());
        out->set_seed(get_seed());
        return out;
    }

private:
    void materialize_in_place() {
        if (materialized_) return;
        // Values without the factor: the factor stays in scalar_ and applies on read.
        const uint64_t n = rows();
        auto m = std::make_unique<SymmetricMatrix<int32_t>>(n, "", sign_ < 0);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = sign_ < 0 ? i + 1 : i; j < n; ++j) m->set(i, j, base_get(i, j));
        }
        materialized_ = std::move(m);
    }

    std::shared_ptr<TriangularMatrix<bool>> bits_;
    std::unique_ptr<SymmetricMatrix<int32_t>> materialized_;
    int sign_;
};

}  // namespace pycauset
