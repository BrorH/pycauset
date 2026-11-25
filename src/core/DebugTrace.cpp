#include "pycauset/core/DebugTrace.hpp"

#include <vector>

namespace pycauset::debug_trace {

thread_local std::string g_last;
thread_local std::string g_last_io;
thread_local bool g_recording = false;
thread_local std::vector<std::string> g_log;

void set_last(const char* value) {
    g_last = value ? value : "";
    if (g_recording) g_log.push_back(g_last);
}

void start_log() {
    g_log.clear();
    g_recording = true;
}

std::vector<std::string> stop_log() {
    g_recording = false;
    return std::move(g_log);
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
