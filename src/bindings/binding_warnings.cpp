#include "binding_warnings.hpp"

#include <mutex>
#include <unordered_set>

namespace pycauset::bindings_warn {

namespace {

struct WarningOnceState {
    std::mutex mutex;
    std::unordered_set<std::string> seen;
};

WarningOnceState& warning_once_state() {
    static WarningOnceState state;
    return state;
}

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

void warn_with_category(const char* message, const char* category_name, int stacklevel) {
    warn_impl(message, category_name, stacklevel);
}

void warn_once(const std::string& key, const std::string& message, int stacklevel) {
    warn_once_with_category(key, message, "PyCausetWarning", stacklevel);
}

void warn_once_with_category(
    const std::string& key,
    const std::string& message,
    const char* category_name,
    int stacklevel) {
    auto& state = warning_once_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.seen.find(key) != state.seen.end()) {
            return;
        }
        state.seen.insert(key);
    }

    warn_impl(message.c_str(), category_name, stacklevel);
}

} // namespace pycauset::bindings_warn
