#include "pycauset/core/DebugTrace.hpp"

namespace pycauset::debug_trace {

thread_local std::string g_last;
thread_local std::string g_last_io;

void set_last(const char* value) {
    g_last = value ? value : "";
}

void clear() {
    g_last.clear();
}

std::string get_last() {
    return g_last;
}

void set_last_io(const char* value) {
    g_last_io = value ? value : "";
}

void clear_io() {
    g_last_io.clear();
}

std::string get_last_io() {
    return g_last_io;
}

} // namespace pycauset::debug_trace
