#pragma once

#include <string>

namespace pycauset::debug_trace {

// Test-only hook used to verify kernel dispatch paths.
// Stored as a thread_local string so concurrent operations don't clobber each other.
void set_last(const char* value);
void clear();
std::string get_last();

} // namespace pycauset::debug_trace
