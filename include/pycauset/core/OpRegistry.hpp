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
    static OpRegistry& instance() {
        static OpRegistry reg;
        return reg;
    }

    void register_op(const OpContract& contract) {
        contracts_[contract.name] = contract;
    }

    const OpContract* get_contract(const std::string& name) const {
        auto it = contracts_.find(name);
        if (it != contracts_.end()) {
            return &it->second;
        }
        return nullptr;
    }

private:
    OpRegistry() = default;
    std::unordered_map<std::string, OpContract> contracts_;
};

} // namespace pycauset
