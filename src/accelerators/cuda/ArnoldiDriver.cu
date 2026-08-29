// ArnoldiDriver
// What: Host-side Arnoldi/Lanczos driver using GPU batch_gemv for matvecs.
// Why: Provides a plug-and-play GPU acceleration routine for top-k eigenvalues.
// Dependencies: CudaDevice batch_gemv and Eigen for Hessenberg eigensolve.

#include "ArnoldiDriver.hpp"
#include "CudaDevice.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/vector/DenseVector.hpp"

#include <random>
#include <algorithm>
#include <complex>
#include <cmath>
#include <stdexcept>

namespace pycauset {

namespace {
    void normalize_vector(std::vector<double>& v) {
        double norm = 0.0;
        for (double x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm == 0.0) return;
        for (double& x : v) x /= norm;
    }

    double dot_ptr(const std::vector<double>& a, const double* b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}

void ArnoldiDriver::run(CudaDevice& device, const MatrixBase& a, VectorBase& out, int k, int m, double tol) {
    if (k <= 0 || m <= 0) {
        throw std::invalid_argument("ArnoldiDriver: k and m must be positive");
    }

    if (a.rows() != a.cols()) {
        throw std::invalid_argument("ArnoldiDriver: matrix must be square");
    }

    auto* out_vec = dynamic_cast<DenseVector<double>*>(&out);
    if (!out_vec) {
        throw std::runtime_error("ArnoldiDriver: output vector must be DenseVector<double>");
    }

    const uint64_t n = a.rows();
    if (static_cast<uint64_t>(k) > n) {
        throw std::invalid_argument("ArnoldiDriver: k cannot exceed matrix size");
    }
    if (out_vec->size() < static_cast<uint64_t>(k)) {
        throw std::invalid_argument("ArnoldiDriver: output vector size too small");
    }

    const int max_steps = std::min<int>(m, static_cast<int>(n));

    std::vector<double> q((max_steps + 1) * n, 0.0);
    std::vector<double> h((max_steps + 1) * max_steps, 0.0);
    std::vector<double> w(n, 0.0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (uint64_t i = 0; i < n; ++i) {
        q[i] = dist(rng);
    }
    normalize_vector(q);

    int steps_completed = 0;
    for (int j = 0; j < max_steps; ++j) {
        const double* qj = &q[j * n];

        std::fill(w.begin(), w.end(), 0.0);
        device.batch_gemv(a, qj, w.data(), 1);

        for (int i = 0; i <= j; ++i) {
            const double* qi = &q[i * n];
            double hij = dot_ptr(w, qi);
            h[i * max_steps + j] = hij;
            for (uint64_t r = 0; r < n; ++r) {
                w[r] -= hij * qi[r];
            }
        }

        double h_next = 0.0;
        for (double val : w) h_next += val * val;
        h_next = std::sqrt(h_next);
        h[(j + 1) * max_steps + j] = h_next;

        if (h_next < tol) {
            steps_completed = j + 1;
            break;
        }

        double* q_next = &q[(j + 1) * n];
        for (uint64_t r = 0; r < n; ++r) {
            q_next[r] = w[r] / h_next;
        }
        steps_completed = j + 1;
    }

    if (steps_completed == 0) {
        throw std::runtime_error("ArnoldiDriver: failed to build Krylov subspace");
    }

    // Build the Hessenberg matrix H (mm x mm) from the Arnoldi recurrence.
    const int mm = steps_completed;
    std::vector<double> A(mm * mm, 0.0);
    for (int i = 0; i < mm; ++i) {
        for (int j = 0; j < mm; ++j) {
            A[i * mm + j] = h[i * max_steps + j];
        }
    }

    // Eigenvalues of the small Hessenberg matrix via explicit unshifted QR
    // iteration (self-contained; replaces the Eigen dependency, which does not
    // compile under nvcc C++20). H is small (Arnoldi subspace), so plain QR
    // converges adequately for the top-k approximation.
    for (int iter = 0; iter < 4000; ++iter) {
        std::vector<double> Q(mm * mm, 0.0);
        std::vector<double> R(mm * mm, 0.0);
        for (int j = 0; j < mm; ++j) {
            std::vector<double> v(mm);
            for (int i = 0; i < mm; ++i) v[i] = A[i * mm + j];
            for (int k = 0; k < j; ++k) {
                double r = 0.0;
                for (int i = 0; i < mm; ++i) r += Q[i * mm + k] * v[i];
                R[k * mm + j] = r;
                for (int i = 0; i < mm; ++i) v[i] -= r * Q[i * mm + k];
            }
            double nrm = 0.0;
            for (double x : v) nrm += x * x;
            nrm = std::sqrt(nrm);
            R[j * mm + j] = nrm;
            for (int i = 0; i < mm; ++i) Q[i * mm + j] = (nrm > 1e-300) ? (v[i] / nrm) : 0.0;
        }
        std::vector<double> A2(mm * mm, 0.0);
        for (int i = 0; i < mm; ++i) {
            for (int j = 0; j < mm; ++j) {
                double s = 0.0;
                for (int k = 0; k < mm; ++k) s += R[i * mm + k] * Q[k * mm + j];
                A2[i * mm + j] = s;
            }
        }
        A.swap(A2);
    }

    // Extract eigenvalues from the (block) upper-triangular real Schur form.
    std::vector<std::pair<double, std::complex<double>>> ranked;
    ranked.reserve(mm);
    for (int i = 0; i < mm; ++i) {
        if (i + 1 < mm && std::abs(A[(i + 1) * mm + i]) > 1e-10) {
            const double a = A[i * mm + i];
            const double b = A[i * mm + i + 1];
            const double c = A[(i + 1) * mm + i];
            const double d = A[(i + 1) * mm + i + 1];
            const double tr = a + d;
            const double disc = (a - d) * (a - d) + 4.0 * b * c;
            if (disc < 0.0) {
                const double re = tr / 2.0;
                const double im = std::sqrt(-disc) / 2.0;
                ranked.emplace_back(std::hypot(re, im), std::complex<double>(re, im));
                ranked.emplace_back(std::hypot(re, im), std::complex<double>(re, -im));
            } else {
                const double sq = std::sqrt(disc);
                ranked.emplace_back(std::abs((tr + sq) / 2.0), std::complex<double>((tr + sq) / 2.0, 0.0));
                ranked.emplace_back(std::abs((tr - sq) / 2.0), std::complex<double>((tr - sq) / 2.0, 0.0));
            }
            ++i;
        } else {
            const double v = A[i * mm + i];
            ranked.emplace_back(std::abs(v), std::complex<double>(v, 0.0));
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& x, const auto& y) {
        return x.first > y.first;
    });

    for (int i = 0; i < k; ++i) {
        if (i >= static_cast<int>(ranked.size())) {
            out_vec->set(i, 0.0);
            continue;
        }
        const auto val = ranked[i].second;
        if (std::abs(val.imag()) > tol) {
            throw std::runtime_error("ArnoldiDriver: complex eigenvalues are not supported in this build");
        }
        out_vec->set(i, val.real());
    }
}

} // namespace pycauset
