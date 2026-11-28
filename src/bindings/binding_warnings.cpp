#include "binding_warnings.hpp"


namespace pycauset::bindings_warn {

namespace {

// The category class from pycauset._internal.warnings. A wrong name is an error, not a silent
// UserWarning (F-101 / PLAN 4.8): the module ships with the package.
py::object resolve_warning_category(const char* category_name) {
    auto mod = py::module_::import("pycauset._internal.warnings");
    return mod.attr(category_name && category_name[0] != '\0' ? category_name : "PyCausetWarning");
}

void warn_impl(const char* message, const char* category_name, int stacklevel) {
    py::module_ warnings = py::module_::import("warnings");
    py::object category = resolve_warning_category(category_name);
    warnings.attr("warn")(py::str(message), category, stacklevel);
}

} // namespace

// Deduplication is Python's job: the default warnings filter shows each warning once per call
// site, and users control it with warnings.simplefilter (a C++ "seen" set would override that).
void warn_with_category(const std::string& message, const char* category_name, int stacklevel) {
    warn_impl(message.c_str(), category_name, stacklevel);
}

} // namespace pycauset::bindings_warn
