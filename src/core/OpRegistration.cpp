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
        eigh.supports_streaming = false;       // Not yet streaming
        eigh.supports_block_matrix = false;    // Dense only usually
        eigh.requires_square = true;
        registry.register_op(eigh);

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
