#pragma once
#include <memory>
#include <string>
#include "MatrixBase.hpp"
#include "TriangularMatrix.hpp"
#include "DenseMatrix.hpp"
#include "VectorBase.hpp"
#include "DenseVector.hpp"

namespace pycauset {

std::unique_ptr<MatrixBase> add(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> subtract(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> elementwise_multiply(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");

std::unique_ptr<MatrixBase> outer_product(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
std::unique_ptr<VectorBase> matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, const std::string& result_file = "");
std::unique_ptr<VectorBase> vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, const std::string& result_file = "");

}

std::unique_ptr<TriangularMatrix<double>> compute_k_matrix(
    const TriangularMatrix<bool>& C, 
    double a, 
    const std::string& output_path, 
    int num_threads = 0
);
