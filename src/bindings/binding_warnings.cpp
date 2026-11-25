#include "binding_warnings.hpp"


namespace pycauset::bindings_warn {

namespace {

py::object resolve_warning_category(const char* category_name) {
    try {
        auto mod = py::module_::import("pycauset._internal.warnings");
        if (category_name && category_name[0] != '\0') {
            return mod.attr(category_name);
        }
        return mod.attr("PyCausetWarning");
    } catch (...) {
        // ignore and fall through
    }

    try {
        auto mod = py::module_::import("pycauset._internal.warnings");
        return mod.attr("PyCausetWarning");
    } catch (...) {
        // ignore and fall through
    }

    return py::module_::import("builtins").attr("UserWarning");
}

void warn_impl(const char* message, const char* category_name, int stacklevel) {
    py::module_ warnings = py::module_::import("warnings");
    py::object category = resolve_warning_category(category_name);
    warnings.attr("warn")(py::str(message), category, stacklevel);
}

} // namespace

void warn(const char* message, int stacklevel) {
    warn_impl(message, "PyCausetWarning", stacklevel);
}

// Deduplication is Python's job: the default warnings filter shows each warning once per call
// site, and users control it with warnings.simplefilter (a C++ "seen" set would override that).
void warn_with_category(const std::string& message, const char* category_name, int stacklevel) {
    warn_impl(message.c_str(), category_name, stacklevel);
}

} // namespace pycauset::bindings_warn
