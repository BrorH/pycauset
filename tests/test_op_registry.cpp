#include <iostream>
#include "pycauset/core/OpRegistry.hpp"

/**
 * @brief Contract Test for R1_CPU Phase 2
 * 
 * Verifies that the OpRegistry can register and retrieve OpContracts.
 */
void test_op_registry() {
    using namespace pycauset;

    auto& registry = OpRegistry::instance();

    // Register a dummy op
    OpContract contract;
    contract.name = "test_op";
    contract.supports_streaming = true;
    contract.supports_block_matrix = false;
    
    registry.register_op(contract);

    // Retrieve it
    const OpContract* fetched = registry.get_contract("test_op");
    
    if (!fetched) {
        std::cerr << "FAILED: Could not retrieve 'test_op'" << std::endl;
        std::exit(1);
    }

    if (fetched->name != "test_op") {
        std::cerr << "FAILED: Name mismatch. Expected 'test_op', got '" << fetched->name << "'" << std::endl;
        std::exit(1);
    }
    
    if (!fetched->supports_streaming) {
        std::cerr << "FAILED: supports_streaming should be true" << std::endl;
        std::exit(1);
    }

    std::cout << "OpRegistry Test Passed." << std::endl;
}

int main() {
    test_op_registry();
    return 0;
}
