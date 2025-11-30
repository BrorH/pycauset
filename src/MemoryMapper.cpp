#include "MemoryMapper.hpp"
#include <iostream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#endif

MemoryMapper::MemoryMapper(const std::string& filename, size_t data_size, bool create_new) 
    : filename_(filename), data_size_(data_size), mapped_ptr_(nullptr) {
    open_file(create_new);
}

MemoryMapper::~MemoryMapper() {
    close_file();
}

void* MemoryMapper::get_data() {
    if (!mapped_ptr_) return nullptr;
    return static_cast<char*>(mapped_ptr_) + sizeof(pycauset::FileHeader);
}

const void* MemoryMapper::get_data() const {
    if (!mapped_ptr_) return nullptr;
    return static_cast<const char*>(mapped_ptr_) + sizeof(pycauset::FileHeader);
}

pycauset::FileHeader* MemoryMapper::get_header() {
    return static_cast<pycauset::FileHeader*>(mapped_ptr_);
}

const pycauset::FileHeader* MemoryMapper::get_header() const {
    return static_cast<const pycauset::FileHeader*>(mapped_ptr_);
}

size_t MemoryMapper::get_data_size() const {
    return data_size_;
}

#ifdef _WIN32
void MemoryMapper::open_file(bool create_new) {
    // Ensure directory exists
    std::filesystem::path path(filename_);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    DWORD creationDisposition = create_new ? CREATE_ALWAYS : OPEN_EXISTING;

    hFile_ = CreateFileA(
        filename_.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, // Allow others to read and write
        NULL,
        creationDisposition,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + filename_);
    }

    size_t total_size = data_size_ + sizeof(pycauset::FileHeader);
    LARGE_INTEGER liSize;
    liSize.QuadPart = total_size;

    // Only set file size if we are creating a new file or explicitly resizing
    if (create_new) {
        if (!SetFilePointerEx(hFile_, liSize, NULL, FILE_BEGIN)) {
            CloseHandle(hFile_);
            throw std::runtime_error("Failed to set file pointer");
        }
        if (!SetEndOfFile(hFile_)) {
            CloseHandle(hFile_);
            throw std::runtime_error("Failed to set end of file");
        }
    } else {
        // Verify file size is at least total_size
        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile_, &fileSize)) {
            CloseHandle(hFile_);
            throw std::runtime_error("Failed to get file size");
        }
        
        // If data_size_ is 0, we map the whole file
        if (data_size_ == 0) {
            total_size = static_cast<size_t>(fileSize.QuadPart);
            if (total_size < sizeof(pycauset::FileHeader)) {
                 CloseHandle(hFile_);
                 throw std::runtime_error("File is too small to contain a header");
            }
            data_size_ = total_size - sizeof(pycauset::FileHeader);
            liSize.QuadPart = total_size;
        } else {
            if (static_cast<size_t>(fileSize.QuadPart) < total_size) {
                 CloseHandle(hFile_);
                 throw std::runtime_error("File is smaller than expected size");
            }
        }
    }

    hMapping_ = CreateFileMappingA(
        hFile_,
        NULL,
        PAGE_READWRITE,
        liSize.HighPart,
        liSize.LowPart,
        NULL
    );

    if (hMapping_ == NULL) {
        CloseHandle(hFile_);
        throw std::runtime_error("Failed to create file mapping");
    }

    mapped_ptr_ = MapViewOfFile(
        hMapping_,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        total_size
    );

    if (mapped_ptr_ == NULL) {
        CloseHandle(hMapping_);
        CloseHandle(hFile_);
        throw std::runtime_error("Failed to map view of file");
    }
}

void MemoryMapper::flush() {
    if (mapped_ptr_) {
        if (!FlushViewOfFile(mapped_ptr_, 0)) {
            // Log warning?
        }
        if (hFile_ != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(hFile_);
        }
    }
}

void MemoryMapper::close_file() {
    if (mapped_ptr_) {
        UnmapViewOfFile(mapped_ptr_);
        mapped_ptr_ = nullptr;
    }
    if (hMapping_) {
        CloseHandle(hMapping_);
        hMapping_ = NULL;
    }
    if (hFile_ != INVALID_HANDLE_VALUE) {
        CloseHandle(hFile_);
        hFile_ = INVALID_HANDLE_VALUE;
    }
}
#else
// POSIX implementation omitted for brevity as we are on Windows, 
// but would follow similar logic using open/ftruncate/mmap.
void MemoryMapper::open_file(bool create_new) {
    throw std::runtime_error("POSIX implementation not updated for new file format");
}
void MemoryMapper::close_file() {}
#endif

