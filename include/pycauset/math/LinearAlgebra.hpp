#pragma once
#include <complex>
#include <memory>
#include <string>

namespace pycauset {

class MatrixBase;
class VectorBase;
template <typename T> class TriangularMatrix;

// --- Matrix Operations ---

std::unique_ptr<MatrixBase> add(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> subtract(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> elementwise_multiply(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> elementwise_divide(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> dispatch_matmul(const MatrixBase& a, const MatrixBase& b, std::string saveas = "");
std::unique_ptr<MatrixBase> cholesky(const MatrixBase& a, const std::string& result_file = "");
std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>> lu(const MatrixBase& a, const std::string& result_file = "");
std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>> qr(const MatrixBase& a, const std::string& result_file = "");
std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> svd(const MatrixBase& a, const std::string& result_file = "");
std::unique_ptr<MatrixBase> solve(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");

std::unique_ptr<VectorBase> eigvals_arnoldi(const MatrixBase& a, int k, int m, double tol, const std::string& result_file = "");
std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eig(const MatrixBase& in, const std::string& result_file = "");
std::unique_ptr<VectorBase> eigvals(const MatrixBase& in, const std::string& result_file = "");
std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eigh(const MatrixBase& in, const std::string& result_file = "");
std::unique_ptr<VectorBase> eigvalsh(const MatrixBase& in, const std::string& result_file = "");


// --- Vector Operations ---

std::unique_ptr<VectorBase> add_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
std::unique_ptr<VectorBase> subtract_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
double dot_product(const VectorBase& a, const VectorBase& b);
std::complex<double> dot_product_complex(const VectorBase& a, const VectorBase& b);
double norm(const VectorBase& v);
double norm(const MatrixBase& m);
std::complex<double> sum(const VectorBase& v);
std::complex<double> sum(const MatrixBase& m);
std::unique_ptr<VectorBase> cross_product(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");

// Scalar operations on vectors
std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, double scalar, const std::string& result_file = "");
std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, int64_t scalar, const std::string& result_file = "");
std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, std::complex<double> scalar, const std::string& result_file = "");
std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, double scalar, const std::string& result_file = "");
std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, int64_t scalar, const std::string& result_file = "");

// --- Matrix-Vector Operations ---

std::unique_ptr<MatrixBase> outer_product(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
std::unique_ptr<VectorBase> matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, const std::string& result_file = "");
std::unique_ptr<VectorBase> vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, const std::string& result_file = "");

}

// --- Special Solvers ---

std::unique_ptr<pycauset::TriangularMatrix<double>> compute_k_matrix(
    const pycauset::TriangularMatrix<bool>& C, 
    double a, 
    const std::string& output_path, 
    int num_threads = 0
);
