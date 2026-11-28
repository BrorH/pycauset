#include "pycauset/compute/ComputeContext.hpp"

namespace pycauset {

ComputeContext& ComputeContext::instance() {
    // Leaked: storage destructors can reach the context during interpreter finalization, where
    // a Meyers singleton's destruction order is undefined.
    static ComputeContext* ctx = new ComputeContext();
    return *ctx;
}

}  // namespace pycauset
