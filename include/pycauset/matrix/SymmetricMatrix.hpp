#pragma once

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"

namespace pycauset {

// Structure bases (see MatrixBase.hpp). The four packed symmetry kinds share one
// implementation (SymmetricMatrix<T> with a runtime kind), so every packed object derives all
// marker bases; the Python type hook picks the class from the object's structure.
//
//   kind             lower triangle A(j, i)   diagonal          dtypes
//   SYMMETRIC        A(i, j)                  any               all
//   ANTISYMMETRIC    -A(i, j)                 0                 signed (not unsigned, not bool)
//   HERMITIAN        conj(A(i, j))            real              complex
//   SKEW_HERMITIAN   -conj(A(i, j))           imaginary         complex
//
// A complex symmetric matrix (A = A^T) and a Hermitian one (A = A^H) are different structures.
class SymmetricMatrixBase : public TriangularMatrixBase {
public:
    using TriangularMatrixBase::TriangularMatrixBase;
};
class AntiSymmetricMatrixBase : public SymmetricMatrixBase {
public:
    using SymmetricMatrixBase::SymmetricMatrixBase;
};
class HermitianMatrixBase : public AntiSymmetricMatrixBase {
public:
    using AntiSymmetricMatrixBase::AntiSymmetricMatrixBase;
};
class SkewHermitianMatrixBase : public HermitianMatrixBase {
public:
    using HermitianMatrixBase::HermitianMatrixBase;
};

inline bool is_symmetric_family(Structure s) {
    return s == Structure::SYMMETRIC || s == Structure::ANTISYMMETRIC || s == Structure::HERMITIAN ||
           s == Structure::SKEW_HERMITIAN;
}

template <typename T>
class SymmetricMatrix : public SkewHermitianMatrixBase {
    static constexpr bool kComplex = is_complex_storage_v<T>;

    static Structure checked(Structure kind) {
        if (!is_symmetric_family(kind)) throw std::invalid_argument("SymmetricMatrix: not a symmetry kind");
        // A(j,i) = -A(i,j) has no meaning for unsigned or bit storage (F-065).
        if ((kind == Structure::ANTISYMMETRIC || kind == Structure::SKEW_HERMITIAN) &&
            (std::is_unsigned_v<T> || std::is_same_v<T, bool>)) {
            throw std::invalid_argument(std::string("antisymmetric matrices need a signed dtype (got ") +
                                        MatrixTraits<T>::name + ")");
        }
        if ((kind == Structure::HERMITIAN || kind == Structure::SKEW_HERMITIAN) && !kComplex) {
            throw std::invalid_argument(std::string("Hermitian matrices need a complex dtype (got ") +
                                        MatrixTraits<T>::name + "); for real dtypes use SymmetricMatrix / "
                                        "AntiSymmetricMatrix");
        }
        return kind;
    }

    static std::string prefix(Structure kind) {
        switch (kind) {
            case Structure::ANTISYMMETRIC: return "antisymmetric_";
            case Structure::HERMITIAN: return "hermitian_";
            case Structure::SKEW_HERMITIAN: return "skew_hermitian_";
            default: return "symmetric_";
        }
    }

public:
    SymmetricMatrix(uint64_t n, const std::string& backing_file = "", Structure kind = Structure::SYMMETRIC)
        : SkewHermitianMatrixBase(n, checked(kind), MatrixTraits<T>::data_type), kind_(kind) {
        has_diagonal_ = true;  // the packed upper triangle includes the diagonal
        uint64_t size_in_bytes = calculate_triangular_offsets(sizeof(T) * 8, 64);
        initialize_storage(size_in_bytes, backing_file, prefix(kind) + MatrixTraits<T>::name, 8, kind,
                           MatrixTraits<T>::data_type, n, n);
    }

    // Constructor for loading with explicit metadata
    SymmetricMatrix(uint64_t n, const std::string& backing_file, size_t offset, uint64_t seed,
                    std::complex<double> scalar, bool is_transposed, Structure kind)
        : SkewHermitianMatrixBase(n, checked(kind), MatrixTraits<T>::data_type), kind_(kind) {
        has_diagonal_ = true;
        uint64_t size_in_bytes = calculate_triangular_offsets(sizeof(T) * 8, 64);
        initialize_storage(size_in_bytes, backing_file, "", 8, kind, MatrixTraits<T>::data_type, n, n, offset,
                           false);
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    SymmetricMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper, Structure kind = Structure::SYMMETRIC)
        : SkewHermitianMatrixBase(n, std::move(mapper), checked(kind), MatrixTraits<T>::data_type), kind_(kind) {
        has_diagonal_ = true;
        calculate_triangular_offsets(sizeof(T) * 8, 64);
    }

    Structure kind() const { return kind_; }
    bool is_antisymmetric() const { return kind_ == Structure::ANTISYMMETRIC; }

    // The value of the lower triangle given the stored upper value (see the table above).
    T mirror(T v) const {
        switch (kind_) {
            case Structure::ANTISYMMETRIC: return negate(v);
            case Structure::HERMITIAN: return conj_of(v);
            case Structure::SKEW_HERMITIAN: return negate(conj_of(v));
            default: return v;
        }
    }

    void set(uint64_t i, uint64_t j, T value) {
        ensure_unique();
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        if (i == j) {
            check_diagonal(value);
        } else if (i > j) {
            std::swap(i, j);
            value = mirror(value);  // the mirror is an involution: storing A(i,j) from A(j,i)
        }
        T* ptr = reinterpret_cast<T*>(static_cast<uint8_t*>(mapper_->get_data()) + get_row_offset(i) +
                                      (j - i) * sizeof(T));
        *ptr = value;
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        const bool lower = i > j;
        if (lower) std::swap(i, j);
        const T* ptr = reinterpret_cast<const T*>(static_cast<const uint8_t*>(mapper_->get_data()) +
                                                  get_row_offset(i) + (j - i) * sizeof(T));
        return lower ? mirror(*ptr) : *ptr;
    }

    T* data() { return static_cast<T*>(this->require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(this->require_mapper()->get_data()); }

    double element_as_double(uint64_t i, uint64_t j) const override {
        if constexpr (kComplex) {
            throw std::runtime_error("Complex matrix does not support get_element_as_double; use get_element_as_complex");
        } else {
            return element_as_complex(i, j).real();
        }
    }

    std::complex<double> element_as_complex(uint64_t i, uint64_t j) const override {
        // The antisymmetric diagonal is +0.0 whatever the factor (F-078).
        if (kind_ == Structure::ANTISYMMETRIC && i == j) return {0.0, 0.0};
        return view_value(get(i, j), scalar_, is_conjugated());
    }

    void set_element_double(uint64_t i, uint64_t j, double value) override {
        if constexpr (kComplex) set(i, j, T(value));
        else set(i, j, static_cast<T>(value));
    }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& = "") const override {
        return scaled_clone(factor);
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& = "") const override {
        return scaled_clone(static_cast<double>(factor));
    }

    std::unique_ptr<MatrixBase> multiply_scalar(std::complex<double> factor, const std::string& = "") const override {
        return scaled_clone(factor);
    }

    // O(1): A^T = A (symmetric), -A (antisymmetric), conj(A) (Hermitian), -conj(A) (skew). A
    // conjugation is applied as a toggled flag with the factor conjugated, so that
    // conj'(raw * s') equals the transposed value under the view rule.
    std::unique_ptr<MatrixBase> transpose(const std::string& = "") const override {
        auto out = std::unique_ptr<MatrixBase>(static_cast<MatrixBase*>(this->clone().release()));
        std::complex<double> s = out->get_scalar();
        if (kind_ == Structure::HERMITIAN || kind_ == Structure::SKEW_HERMITIAN) {
            out->set_conjugated(!out->is_conjugated());
            s = std::conj(s);
        }
        if (kind_ == Structure::ANTISYMMETRIC || kind_ == Structure::SKEW_HERMITIAN) s = -s;
        out->set_scalar(s);
        return out;
    }

    std::unique_ptr<PersistentObject> clone() const override {
        // Keep the view metadata: transpose/negation and scaling are stored in it.
        auto out = std::make_unique<SymmetricMatrix<T>>(base_rows(), mapper_, kind_);
        out->set_scalar(this->get_scalar());
        out->set_conjugated(this->is_conjugated());
        out->set_transposed(this->is_transposed());
        out->set_seed(this->get_seed());
        return out;
    }

    // The matrix of the given kind whose upper triangle (strict for antisymmetric) is `source`'s.
    static std::unique_ptr<SymmetricMatrix<T>> from_triangular(const TriangularMatrix<T>& source, Structure kind,
                                                                const std::string& backing_file = "") {
        const uint64_t n = source.rows();
        auto mat = std::make_unique<SymmetricMatrix<T>>(n, backing_file, kind);
        const bool skip_diag = kind == Structure::ANTISYMMETRIC;
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = skip_diag ? i + 1 : i; j < n; ++j) mat->set(i, j, source.get(i, j));
        }
        return mat;
    }

private:
    static T negate(T v) {
        if constexpr (std::is_same_v<T, bool>) return v;  // never reached: bool is never antisymmetric
        else return static_cast<T>(-v);
    }
    static T conj_of(T v) {
        if constexpr (kComplex) {
            using std::conj;  // std::complex; complex_half's conj is found by argument lookup
            return T(conj(v));
        } else {
            return v;
        }
    }
    void check_diagonal(T v) const {
        if (kind_ == Structure::ANTISYMMETRIC && v != T(0)) {
            throw std::invalid_argument("Diagonal of an antisymmetric matrix must be zero");
        }
        if constexpr (kComplex) {
            if (kind_ == Structure::HERMITIAN && v.imag() != 0) {
                throw std::invalid_argument("Diagonal of a Hermitian matrix must be real");
            }
            if (kind_ == Structure::SKEW_HERMITIAN && v.real() != 0) {
                throw std::invalid_argument("Diagonal of a skew-Hermitian matrix must be imaginary");
            }
        }
    }

    Structure kind_;
};

}  // namespace pycauset
