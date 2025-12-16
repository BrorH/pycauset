#include "pycauset/core/DebugTrace.hpp"

namespace pycauset::debug_trace {

thread_local std::string g_last;

void set_last(const char* value) {
    g_last = value ? value : "";
}

void clear() {
    g_last.clear();
}

std::string get_last() {
    return g_last;
}

} // namespace pycauset::debug_trace
