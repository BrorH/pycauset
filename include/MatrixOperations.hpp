#pragma once
#include <memory>
#include <string>
#include "MatrixBase.hpp"
#include "TriangularMatrix.hpp"
#include "DenseMatrix.hpp"

namespace pycauset {

std::unique_ptr<MatrixBase> add(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> subtract(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> elementwise_multiply(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");

}

void compute_k_matrix(
    const TriangularMatrix<bool>& C, 
    double a, 
    const std::string& output_path, 
    int num_threads = 0
);
