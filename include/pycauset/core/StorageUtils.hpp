#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>

#include "pycauset/core/Types.hpp"

namespace pycauset {

// --- Storage Configuration (from StoragePaths.hpp) ---

// Global configuration for memory threshold
void set_memory_threshold(uint64_t bytes);
uint64_t get_memory_threshold();

// A threshold for objects created on the calling thread only (-1: none, the global one
// applies). Per-call placement (`storage="disk"`, `max_in_ram_bytes`) used to set the global
// threshold for the call, so other threads' objects went to files meanwhile (review 3.8 C-5).
// Returns the previous override, to restore.
int64_t set_thread_memory_threshold(int64_t bytes);
// The threshold that places an object created on this thread.
uint64_t placement_threshold();

std::filesystem::path get_storage_root();
void set_storage_root(const std::filesystem::path& root);
std::string make_unique_storage_file(const std::string& prefix);

}
