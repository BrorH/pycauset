#pragma once

#include <cstdint>
#include <string>

namespace pycauset {

class SystemUtils {
public:

    /**
     * @brief Get the current user's home directory (best-effort).
     * 
     * @return std::string Home directory path, or empty if unavailable.
     */
    static std::string get_home_directory();
};

} // namespace pycauset
