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

uint64_t SystemUtils::get_available_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        return statex.ullAvailPhys;
    }
    return 0;
#elif defined(__APPLE__)
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    vm_statistics_data_t vm_stat;
    if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vm_stat, &count) == KERN_SUCCESS) {
        return (uint64_t)vm_stat.free_count * sysconf(_SC_PAGESIZE);
    }
    return 0;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
    return 0;
#endif
}

uint64_t SystemUtils::get_total_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        return statex.ullTotalPhys;
    }
    return 0;
#elif defined(__APPLE__)
    int mib[2];
    int64_t physical_memory;
    size_t length;
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(int64_t);
    if (sysctl(mib, 2, &physical_memory, &length, NULL, 0) == 0) {
        return physical_memory;
    }
    return 0;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
    return 0;
#endif
}

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
