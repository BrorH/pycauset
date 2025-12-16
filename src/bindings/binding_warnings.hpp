#pragma once

#include "bindings_common.hpp"

#include <string>

namespace pycauset::bindings_warn {

// Emit a warning with PyCauset's default warning category when available.
void warn(const char* message, int stacklevel = 2);

// Emit a warning with an explicit category name from pycauset._internal.warnings,
// falling back to PyCausetWarning and then UserWarning.
void warn_with_category(const char* message, const char* category_name, int stacklevel = 2);

// Warn-once variants keyed by a stable string.
void warn_once(const std::string& key, const std::string& message, int stacklevel = 2);
void warn_once_with_category(
    const std::string& key,
    const std::string& message,
    const char* category_name,
    int stacklevel = 2);

} // namespace pycauset::bindings_warn
