#include "pycauset/math/Eigen.hpp"

#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace pycauset {

namespace {
std::vector<double> to_memory_flat_real(const MatrixBase& m) {
    uint64_t n = m.size();
    std::vector<double> mat(n * n);

    if (auto* dm = dynamic_cast<const DenseMatrix<double>*>(&m)) {
        const double* src = dm->data();
        std::copy(src, src + n * n, mat.begin());
        return mat;
    }

    if (auto* fm = dynamic_cast<const DenseMatrix<float>*>(&m)) {
        const float* src = fm->data();
        std::transform(src, src + n * n, mat.begin(), [](float v) { return static_cast<double>(v); });
        return mat;
    }

    ParallelFor(0, n, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            mat[i * n + j] = m.get_element_as_double(i, j);
        }
    });

    return mat;
}

std::vector<std::complex<double>> to_memory_flat_complex(const MatrixBase& m) {
    uint64_t n = m.size();
    std::vector<std::complex<double>> mat(n * n);

    ParallelFor(0, n, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            mat[i * n + j] = m.get_element_as_complex(i, j);
        }
    });

    return mat;
}
} // namespace

double trace(const MatrixBase& matrix) {
    if (auto cached = matrix.get_cached_trace()) {
        return *cached;
    }

    const uint64_t n = matrix.size();
    const auto type = matrix.get_matrix_type();

    double tr = 0.0;
    if (type == MatrixType::IDENTITY) {
        tr = matrix.get_scalar().real() * static_cast<double>(n);
    } else {
        for (uint64_t i = 0; i < n; ++i) {
            tr += matrix.get_element_as_double(i, i);
        }
    }

    matrix.set_cached_trace(tr);
    return tr;
}

double determinant(const MatrixBase& matrix) {
    if (auto cached = matrix.get_cached_determinant()) {
        return *cached;
    }

    const uint64_t n = matrix.size();
    const auto type = matrix.get_matrix_type();

    double det = 0.0;

    if (type == MatrixType::IDENTITY) {
        det = std::pow(matrix.get_scalar(), n).real();
        matrix.set_cached_determinant(det);
        return det;
    }

    if (type == MatrixType::DIAGONAL) {
        det = 1.0;
        for (uint64_t i = 0; i < n; ++i) {
            det *= matrix.get_element_as_double(i, i);
        }
        matrix.set_cached_determinant(det);
        return det;
    }

    if (type == MatrixType::TRIANGULAR_FLOAT || type == MatrixType::CAUSAL) {
        bool has_diag = false;
        if (auto* m = dynamic_cast<const TriangularMatrix<double>*>(&matrix)) {
            has_diag = m->has_diagonal();
        } else if (auto* m = dynamic_cast<const TriangularMatrix<int32_t>*>(&matrix)) {
            has_diag = m->has_diagonal();
        }

        if (!has_diag) {
            det = 0.0;
            matrix.set_cached_determinant(det);
            return det;
        }

        det = 1.0;
        for (uint64_t i = 0; i < n; ++i) {
            det *= matrix.get_element_as_double(i, i);
        }

        matrix.set_cached_determinant(det);
        return det;
    }

    // General case: LU determinant.
    // If there is any complex scalar/component, compute using complex LU and return the real part.
    const bool use_complex = std::abs(matrix.get_scalar().imag()) > 1e-14;

    if (!use_complex) {
        auto data = to_memory_flat_real(matrix);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_eigen(
            data.data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

        Eigen::PartialPivLU<Eigen::MatrixXd> lu(mat_eigen);
        det = lu.determinant();
    } else {
        auto data = to_memory_flat_complex(matrix);
        Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_eigen(
            data.data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(mat_eigen);
        det = lu.determinant().real();
    }

    matrix.set_cached_determinant(det);
    return det;
}

} // namespace pycauset
