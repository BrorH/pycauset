#pragma once

#include "bindings_common.hpp"

#include <string>

namespace pycauset::bindings_warn {

// Emit a warning with PyCauset's default warning category when available.

void warn_with_category(const std::string& message, const char* category_name, int stacklevel = 2);

} // namespace pycauset::bindings_warn
