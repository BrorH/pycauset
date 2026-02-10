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
        eigh.supports_streaming = false;       // Not yet streaming (Phase 6: full matrix required in memory)
        eigh.supports_block_matrix = false;    // Dense only - LAPACK requires contiguous full matrix
        eigh.requires_square = true;
        registry.register_op(eigh);

        // --- Eigvalsh (Eigenvalues only for Hermitian) ---
        OpContract eigvalsh;
        eigvalsh.name = "eigvalsh";
        eigvalsh.supports_streaming = false;   // LAPACK dsyev/cheev requires full matrix
        eigvalsh.supports_block_matrix = false; // Dense only - no block decomposition for eigenvalues
        eigvalsh.requires_square = true;
        registry.register_op(eigvalsh);

        // --- Eig (General Eigenvalues/Vectors) ---
        OpContract eig;
        eig.name = "eig";
        eig.supports_streaming = false;        // LAPACK dgeev requires full matrix
        eig.supports_block_matrix = false;     // Dense only - no block support for general eigen
        eig.requires_square = true;
        registry.register_op(eig);

        // --- Eigvals (General Eigenvalues only) ---
        OpContract eigvals;
        eigvals.name = "eigvals";
        eigvals.supports_streaming = false;    // LAPACK dgeev requires full matrix
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
