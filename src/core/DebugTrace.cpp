#include "pycauset/core/DebugTrace.hpp"

#include <vector>

namespace pycauset::debug_trace {

thread_local std::string g_last;
thread_local std::string g_last_io;
thread_local std::vector<std::string> g_log;
// Logs nest (the atlas logs around an operation that writes its own IO trace, F-100): each
// start_log() marks where its entries begin; stop_log() returns them and leaves the outer
// log's entries in place.
thread_local std::vector<size_t> g_marks;

void set_last(const char* value) {
    g_last = value ? value : "";
    if (!g_marks.empty()) g_log.push_back(g_last);
}

void start_log() {
    g_marks.push_back(g_log.size());
}

std::vector<std::string> stop_log() {
    if (g_marks.empty()) return {};
    const size_t mark = g_marks.back();
    g_marks.pop_back();
    std::vector<std::string> out(g_log.begin() + static_cast<std::ptrdiff_t>(mark), g_log.end());
    if (g_marks.empty()) g_log.clear();
    return out;
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

#include "pycauset/core/AccessCounter.hpp"

namespace pycauset::access_counter {

std::atomic<bool> g_enabled{false};
std::atomic<uint64_t> g_count{0};

} // namespace pycauset::access_counter
