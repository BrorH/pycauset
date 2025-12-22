#pragma once

#include <string>

namespace pycauset::debug_trace {

// Test-only hook used to verify kernel dispatch paths.
// Stored as a thread_local string so concurrent operations don't clobber each other.
void set_last(const char* value);
void clear();
std::string get_last();

// Separate IO accelerator trace channel so IO hints don't clobber kernel traces.
void set_last_io(const char* value);
void clear_io();
std::string get_last_io();

} // namespace pycauset::debug_trace
