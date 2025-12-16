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
std::unique_ptr<MatrixBase> dispatch_matmul(const MatrixBase& a, const MatrixBase& b, std::string saveas = "");

// --- Vector Operations ---

std::unique_ptr<VectorBase> add_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
std::unique_ptr<VectorBase> subtract_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
double dot_product(const VectorBase& a, const VectorBase& b);
std::complex<double> dot_product_complex(const VectorBase& a, const VectorBase& b);
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
