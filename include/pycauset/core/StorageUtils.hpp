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

std::filesystem::path get_storage_root();
void set_storage_root(const std::filesystem::path& root);
std::string make_unique_storage_file(const std::string& prefix);

}
