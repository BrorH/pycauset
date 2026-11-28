#pragma once
#include <complex>
#include <cstdint>
#include <memory>
#include <string>

#include "pycauset/core/Types.hpp"

namespace pycauset {

class MatrixBase;
class VectorBase;
template <typename T> class TriangularMatrix;

// NumPy's LAPACK dtype for operands of dtypes a and b: integers and bit in float64, float16 /
// float32 in float32, complex stays complex (complex64 when nothing needs double precision).
DataType lapack_dtype(DataType a, DataType b);

// --- Matrix Operations ---

std::unique_ptr<MatrixBase> add(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> subtract(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> elementwise_multiply(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> elementwise_divide(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
std::unique_ptr<MatrixBase> dispatch_matmul(const MatrixBase& a, const MatrixBase& b, std::string saveas = "");

// Matrix inverse (NumPy inv): solve(A, I) in the LAPACK dtype; structure rules for identity,
// diagonal and triangular input. Raises std::runtime_error when singular.
std::unique_ptr<MatrixBase> inverse(const MatrixBase& a);
std::unique_ptr<MatrixBase> cholesky(const MatrixBase& a, const std::string& result_file = "");
std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>> lu(const MatrixBase& a);
// NumPy's qr: reduced (Q m x k, R k x n) or complete (Q m x m, R m x n).
std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<MatrixBase>> qr(const MatrixBase& a, bool complete = false);
// NumPy's svd: any dtype/structure/view; U (m, m|k), S (k,) real, VT (n|k, n). Without
// compute_uv only S is computed (U is m x 0 and VT 0 x n).
std::tuple<std::unique_ptr<MatrixBase>, std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> svd(
    const MatrixBase& a, bool full_matrices = false, bool compute_uv = true);
std::unique_ptr<MatrixBase> solve(const MatrixBase& a, const MatrixBase& b, const std::string& result_file = "");
// NumPy's lstsq solution: x minimizing ||a x - b||, minimum norm when rank-deficient (?gelsd,
// rcond = eps * max(m, n)). Any dtype/structure/view; x is n x b.cols().
std::unique_ptr<MatrixBase> lstsq(const MatrixBase& a, const MatrixBase& b);
// NumPy's pinv (default rtol): through svd, any dtype/structure/view.
std::unique_ptr<MatrixBase> pinv(const MatrixBase& a);

std::unique_ptr<VectorBase> eigvals_arnoldi(const MatrixBase& a, int k, int m, double tol, const std::string& result_file = "");
std::unique_ptr<VectorBase> eigvals_skew(const MatrixBase& a, int k, const std::string& result_file = "");
std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eig_skew(const MatrixBase& a, int k, const std::string& result_file = "");
std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eig(const MatrixBase& in);
std::unique_ptr<VectorBase> eigvals(const MatrixBase& in);
std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eigh(const MatrixBase& in);
std::unique_ptr<VectorBase> eigvalsh(const MatrixBase& in);


// --- Vector Operations ---

// a . b (orientation ignored, no conjugation, as np.dot) as a 1 x 1 matrix of matmul's result
// dtype: the 1-D case of matmul (integers exact, overflow raises; bool counts).
std::unique_ptr<MatrixBase> dot(const VectorBase& a, const VectorBase& b);
double norm(const VectorBase& v);
double norm(const MatrixBase& m);
// NumPy's ord-norms: matrix ord in {"fro", "nuc", "1", "-1", "2", "-2", "inf", "-inf"}; vector p
// any real or +-inf. matrix_rank: tol < 0 is NumPy's default tolerance.
double matrix_norm(const MatrixBase& a, const std::string& ord);
double vector_norm(const VectorBase& v, double p);
int64_t matrix_rank(const MatrixBase& a, double tol = -1.0);
std::unique_ptr<VectorBase> cross_product(const VectorBase& a, const VectorBase& b);

// Scalar operations on vectors

// --- Matrix-Vector Operations ---

std::unique_ptr<MatrixBase> outer_product(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
std::unique_ptr<VectorBase> matrix_vector_multiply(const MatrixBase& m, const VectorBase& v);
std::unique_ptr<VectorBase> vector_matrix_multiply(const VectorBase& v, const MatrixBase& m);

}

// --- Special Solvers ---

// K = C (aI + C)^-1 for a causal matrix C (TriangularMatrix of dtype bool, no diagonal). A
// transposed C gives K(C)^T (the inverse commutes with C). a == 0: linalg_error (singular).
std::unique_ptr<pycauset::MatrixBase> compute_k_matrix(const pycauset::MatrixBase& C, double a);
