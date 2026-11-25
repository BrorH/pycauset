// Typed readers for every matrix class with the view applied once (review PLAN N.2).
// Kept in its own translation unit: one instantiation per (class, wide type) pair.
#include "pycauset/math/ViewRead.hpp"

#include <string>
#include <type_traits>

#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"

namespace pycauset {
namespace {

template <typename T>
std::complex<double> view_value(const T& raw, std::complex<double> s, bool conj) {
    std::complex<double> z;
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        if (s == std::complex<double>(1.0, 0.0)) z = {static_cast<double>(raw), 0.0};
        else if (scalar::is_exact_int_factor(s))
            z = {static_cast<double>(scalar::wrap_scale(raw, static_cast<int64_t>(s.real()))), 0.0};
        else z = static_cast<double>(raw) * s;
    } else if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, bool> ||
                         std::is_same_v<T, pycauset::float16_t>) {
        // Real storage, real factor: real arithmetic, as NumPy scales a real array. A factor
        // stored as (-1, -0) (A.T of an antisymmetric A) must give 0.0 * -1 = -0.0; a complex
        // product would give +0.0 (found by the atlas, 2 / A.T).
        const double r = scalar::to_complex_double(raw).real();
        z = s.imag() == 0.0 ? std::complex<double>(r * s.real(), 0.0) : r * s;
    } else {
        z = scalar::to_complex_double(raw);
        if (s != std::complex<double>(1.0, 0.0)) z *= s;
    }
    return conj ? std::conj(z) : z;
}

template <typename W, typename Src>
std::unique_ptr<MatrixBase> read_typed(const Src& m, const MatrixBase& a, bool anti) {
    const std::complex<double> s = a.get_scalar();
    const bool conj = a.is_conjugated();
    const uint64_t rows = a.rows(), cols = a.cols();
    auto out = std::make_unique<DenseMatrix<W>>(rows, cols, std::string());
    W* dst = out->data();
    ParallelFor(0, rows, [&](size_t ii) {
        const uint64_t i = static_cast<uint64_t>(ii);
        for (uint64_t j = 0; j < cols; ++j) {
            W v;
            if (anti && i == j) {
                v = W(0);
            } else {
                const auto raw = m.get(i, j);
                using In = std::decay_t<decltype(raw)>;
                if constexpr (std::is_same_v<W, int64_t> && std::is_integral_v<In> && !std::is_same_v<In, bool>) {
                    // Integer storage into an integer result stays exact (no detour through double).
                    if (scalar::is_exact_int_factor(s)) {
                        const int64_t k = static_cast<int64_t>(s.real());
                        v = static_cast<int64_t>(k == 1 ? raw : scalar::wrap_scale(raw, k));
                    } else {
                        v = static_cast<int64_t>(view_value(raw, s, conj).real());
                    }
                } else {
                    const std::complex<double> z = view_value(raw, s, conj);
                    if constexpr (std::is_same_v<W, std::complex<double>>) v = z;
                    else if constexpr (std::is_same_v<W, double>) v = z.real();
                    else v = static_cast<int64_t>(z.real());
                }
            }
            dst[i * cols + j] = v;
        }
    });
    return out;
}

template <typename W>
std::unique_ptr<MatrixBase> read_as(const MatrixBase& a) {
    const bool window = a.is_subview();
#define PYCAUSET_READ(Cls, anti) \
    if (auto* m = dynamic_cast<const Cls*>(&a)) return read_typed<W>(*m, a, anti);
#define PYCAUSET_READ_REAL(Tmpl, anti)                                                                    \
    PYCAUSET_READ(Tmpl<bool>, anti) PYCAUSET_READ(Tmpl<int8_t>, anti) PYCAUSET_READ(Tmpl<int16_t>, anti)     \
    PYCAUSET_READ(Tmpl<int32_t>, anti) PYCAUSET_READ(Tmpl<int64_t>, anti) PYCAUSET_READ(Tmpl<uint8_t>, anti) \
    PYCAUSET_READ(Tmpl<uint16_t>, anti) PYCAUSET_READ(Tmpl<uint32_t>, anti)                                  \
    PYCAUSET_READ(Tmpl<uint64_t>, anti) PYCAUSET_READ(Tmpl<float>, anti) PYCAUSET_READ(Tmpl<double>, anti)   \
    PYCAUSET_READ(Tmpl<pycauset::float16_t>, anti)
    // Dense `get` honours windows and transposition (the bit specialisation ignores windows).
    PYCAUSET_READ(DenseMatrix<int8_t>, false) PYCAUSET_READ(DenseMatrix<int16_t>, false)
    PYCAUSET_READ(DenseMatrix<int32_t>, false) PYCAUSET_READ(DenseMatrix<int64_t>, false)
    PYCAUSET_READ(DenseMatrix<uint8_t>, false) PYCAUSET_READ(DenseMatrix<uint16_t>, false)
    PYCAUSET_READ(DenseMatrix<uint32_t>, false) PYCAUSET_READ(DenseMatrix<uint64_t>, false)
    PYCAUSET_READ(DenseMatrix<float>, false) PYCAUSET_READ(DenseMatrix<double>, false)
    PYCAUSET_READ(DenseMatrix<pycauset::float16_t>, false)
    PYCAUSET_READ(DenseMatrix<std::complex<float>>, false)
    PYCAUSET_READ(DenseMatrix<std::complex<double>>, false)
    if (window) return nullptr;
    // The other classes' `get` honours transposition (or is symmetric), not windows.
    PYCAUSET_READ(DenseMatrix<bool>, false)
    PYCAUSET_READ(ComplexFloat16Matrix, false)
    if (a.get_structure() == Structure::SYMMETRIC || a.get_structure() == Structure::ANTISYMMETRIC) {
        const bool anti = a.get_structure() == Structure::ANTISYMMETRIC;
        PYCAUSET_READ_REAL(SymmetricMatrix, anti)
    }
    PYCAUSET_READ_REAL(TriangularMatrix, false)
    PYCAUSET_READ_REAL(DiagonalMatrix, false)
    PYCAUSET_READ_REAL(IdentityMatrix, false)
#undef PYCAUSET_READ_REAL
#undef PYCAUSET_READ
    return nullptr;
}

}  // namespace

std::unique_ptr<MatrixBase> read_wide(const MatrixBase& a, WideKind kind) {
    switch (kind) {
        case WideKind::Int64: return read_as<int64_t>(a);
        case WideKind::Float64: return read_as<double>(a);
        case WideKind::Complex128: return read_as<std::complex<double>>(a);
    }
    return nullptr;
}

}  // namespace pycauset
