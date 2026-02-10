#include "pycauset/core/OpRegistry.hpp"

namespace pycauset {

// This file registers the standard operations with the OpRegistry.
// It is linked into the core library and runs at static initialization time.

namespace {

struct OpRegistration {
    OpRegistration() {
        auto& registry = OpRegistry::instance();

        // --- Matmul ---
        OpContract matmul;
        matmul.name = "matmul";
        matmul.supports_streaming = true;      // Controlled by StreamingManager / AutoSolver
        matmul.supports_block_matrix = true;   // Recursive block matrix support
        matmul.requires_square = false;
        registry.register_op(matmul);

        // --- Inverse ---
        OpContract inverse;
        inverse.name = "inverse";
        inverse.supports_streaming = true;     // Block-recursive inverse
        inverse.supports_block_matrix = true;
        inverse.requires_square = true;
        registry.register_op(inverse);

        // --- Cholesky ---
        OpContract cholesky;
        cholesky.name = "cholesky";
        cholesky.supports_streaming = true;    // Tiled Cholesky
        cholesky.supports_block_matrix = true;
        cholesky.requires_square = true;
        registry.register_op(cholesky);

        // --- Eigh (Eigenvalues/Vectors for Hermitian) ---
        OpContract eigh;
        eigh.name = "eigh";
        eigh.supports_streaming = false;       // LAPACK architectural constraint: requires full contiguous matrix
        eigh.supports_block_matrix = false;    // Dense only - LAPACK requires contiguous full matrix
        eigh.requires_square = true;
        registry.register_op(eigh);

        // --- Eigvalsh (Eigenvalues only for Hermitian) ---
        OpContract eigvalsh;
        eigvalsh.name = "eigvalsh";
        eigvalsh.supports_streaming = false;   // LAPACK dsyev/cheev architectural constraint
        eigvalsh.supports_block_matrix = false; // Dense only - no block decomposition for eigenvalues
        eigvalsh.requires_square = true;
        registry.register_op(eigvalsh);

        // --- Eig (General Eigenvalues/Vectors) ---
        OpContract eig;
        eig.name = "eig";
        eig.supports_streaming = false;        // LAPACK dgeev architectural constraint
        eig.supports_block_matrix = false;     // Dense only - no block support for general eigen
        eig.requires_square = true;
        registry.register_op(eig);

        // --- Eigvals (General Eigenvalues only) ---
        OpContract eigvals;
        eigvals.name = "eigvals";
        eigvals.supports_streaming = false;    // LAPACK dgeev architectural constraint
        eigvals.supports_block_matrix = false; // Dense only
        eigvals.requires_square = true;
        registry.register_op(eigvals);

        // --- Eigvals Arnoldi (Top-k Eigenvalues) ---
        OpContract eigvals_arnoldi;
        eigvals_arnoldi.name = "eigvals_arnoldi";
        eigvals_arnoldi.supports_streaming = true;  // Arnoldi can work with matrix-vector products (out-of-core A)
        eigvals_arnoldi.supports_block_matrix = false; // Currently dense only, but A can be disk-backed
        eigvals_arnoldi.requires_square = true;
        registry.register_op(eigvals_arnoldi);

        // --- Trace ---
        OpContract trace;
        trace.name = "trace";
        trace.supports_streaming = true;   // Streaming-safe: only diagonal access
        trace.supports_block_matrix = true; // Can sum diagonals from blocks
        trace.requires_square = false;     // Works on non-square (trace of min(m,n) diagonal)
        registry.register_op(trace);

        // --- Determinant ---
        OpContract determinant;
        determinant.name = "determinant";
        determinant.supports_streaming = false; // Uses LU decomposition (requires full matrix)
        determinant.supports_block_matrix = true; // Can use block determinant formula
        determinant.requires_square = true;
        registry.register_op(determinant);

        // --- Norm Operations ---
        OpContract frobenius_norm;
        frobenius_norm.name = "frobenius_norm";
        frobenius_norm.supports_streaming = true;  // Sum of squares - streaming-safe
        frobenius_norm.supports_block_matrix = true; // Can sum block norms
        frobenius_norm.requires_square = false;
        registry.register_op(frobenius_norm);

        // --- Linear Algebra Factorizations ---
        OpContract qr;
        qr.name = "qr";
        qr.supports_streaming = false;  // LAPACK geqrf requires full matrix
        qr.supports_block_matrix = false; // Dense only
        qr.requires_square = false;
        registry.register_op(qr);

        OpContract lu;
        lu.name = "lu";
        lu.supports_streaming = false;  // LAPACK getrf requires full matrix
        lu.supports_block_matrix = false; // Dense only
        lu.requires_square = true;  // Current implementation requires square (PA=LU)
        registry.register_op(lu);

        OpContract svd;
        svd.name = "svd";
        svd.supports_streaming = false;  // LAPACK gesdd requires full matrix
        svd.supports_block_matrix = false; // Dense only
        svd.requires_square = false;
        registry.register_op(svd);

        OpContract solve;
        solve.name = "solve";
        solve.supports_streaming = false;  // LAPACK getrf/getrs requires full matrix
        solve.supports_block_matrix = false; // Dense only (A must be dense)
        solve.requires_square = true;  // A must be square to solve Ax=b
        registry.register_op(solve);

        // --- Elementwise ---
        OpContract add;
        add.name = "add";
        add.supports_streaming = true;
        add.supports_block_matrix = true;
        add.requires_square = false;
        registry.register_op(add);

        OpContract subtract;
        subtract.name = "subtract";
        subtract.supports_streaming = true;
        subtract.supports_block_matrix = true;
        subtract.requires_square = false;
        registry.register_op(subtract);

        OpContract multiply;
        multiply.name = "multiply";
        multiply.supports_streaming = true;
        multiply.supports_block_matrix = true;
        multiply.requires_square = false;
        registry.register_op(multiply);

        OpContract divide;
        divide.name = "divide";
        divide.supports_streaming = true;
        divide.supports_block_matrix = true;
        divide.requires_square = false;
        registry.register_op(divide);
    }
};

static OpRegistration global_op_registration;

} // namespace
} // namespace pycauset
