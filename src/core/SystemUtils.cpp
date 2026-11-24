#include "pycauset/core/SystemUtils.hpp"
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <unistd.h>
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

namespace pycauset {

std::string SystemUtils::get_home_directory() {
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
    if (home && *home) {
        return std::string(home);
    }
    const char* drive = std::getenv("HOMEDRIVE");
    const char* path = std::getenv("HOMEPATH");
    if (drive && path) {
        return std::string(drive) + std::string(path);
    }
    return std::string();
#else
    const char* home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home);
    }
    return std::string();
#endif
}

} // namespace pycauset
