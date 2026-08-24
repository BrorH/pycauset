#pragma once

#include <string>
#include <functional>
#include <unordered_map>
#include <memory>
#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

struct OpContract {
    std::string name;
    bool supports_streaming = false;
    bool supports_block_matrix = false;
    bool requires_square = false;
    // Add more contract fields as needed (e.g., SIMD tiers, property rules)
};

class OpRegistry {
public:
    // NOTE: defined out-of-line in src/core/OpRegistration.cpp so that the
    // function-local static registry is a SINGLE object shared across DLLs
    // (pycauset_core.dll registers; _pycauset.pyd reads). An inline definition
    // here would give every DLL its own (empty) registry.
    static OpRegistry& instance();
    void register_op(const OpContract& contract);
    const OpContract* get_contract(const std::string& name) const;

private:
    OpRegistry() = default;
    std::unordered_map<std::string, OpContract> contracts_;
};

} // namespace pycauset
