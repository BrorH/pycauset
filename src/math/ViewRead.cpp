// Typed readers for every matrix class with the view applied once (review PLAN N.2).
// Kept in its own translation unit: one instantiation per (class, wide type) pair.
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/core/DTypeDispatch.hpp"

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
    // Dense `get` honours windows and transposition; the other classes' `get` (and the bit
    // specialisation's) honour transposition, not windows: their windows read virtually.
    const Structure st = a.get_structure();
    if (a.is_subview() && (st != Structure::DENSE || a.get_data_type() == DataType::BIT)) return nullptr;
    // A lazy closure computes on read: it is evaluated by the virtual path (or materialised).
    if (dynamic_cast<const BitClosureMatrix*>(&a) != nullptr) return nullptr;
    // One class dispatch through the catalogue (DTypeDispatch.hpp): every offered class.
    return visit_matrix(a, [&](const auto& typed) -> std::unique_ptr<MatrixBase> {
        return read_typed<W>(typed, a, st == Structure::ANTISYMMETRIC);
    });
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
