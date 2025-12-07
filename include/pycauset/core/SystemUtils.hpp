#pragma once

#include <cstdint>

namespace pycauset {

class SystemUtils {
public:
    /**
     * @brief Get the total available physical memory (RAM) in bytes.
     * 
     * @return uint64_t Available RAM in bytes.
     */
    static uint64_t get_available_ram();

    /**
     * @brief Get the total physical memory (RAM) installed in the system in bytes.
     * 
     * @return uint64_t Total RAM in bytes.
     */
    static uint64_t get_total_ram();
};

} // namespace pycauset
