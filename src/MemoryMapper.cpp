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

MemoryMapper::MemoryMapper(const std::string& filename, size_t size) 
    : filename_(filename), size_(size), data_(nullptr) {
    openFile();
}

MemoryMapper::~MemoryMapper() {
    closeFile();
}

void* MemoryMapper::getData() {
    return data_;
}

const void* MemoryMapper::getData() const {
    return data_;
}

size_t MemoryMapper::getSize() const {
    return size_;
}

#ifdef _WIN32
void MemoryMapper::openFile() {
    // Ensure directory exists
    std::filesystem::path path(filename_);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    hFile_ = CreateFileA(
        filename_.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0, // No sharing
        NULL,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + filename_);
    }

    // Set file size
    LARGE_INTEGER liSize;
    liSize.QuadPart = size_;
    if (!SetFilePointerEx(hFile_, liSize, NULL, FILE_BEGIN)) {
        CloseHandle(hFile_);
        throw std::runtime_error("Failed to set file pointer");
    }
    if (!SetEndOfFile(hFile_)) {
        CloseHandle(hFile_);
        throw std::runtime_error("Failed to set end of file");
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

    data_ = MapViewOfFile(
        hMapping_,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        size_
    );

    if (data_ == NULL) {
        CloseHandle(hMapping_);
        CloseHandle(hFile_);
        throw std::runtime_error("Failed to map view of file");
    }
}

void MemoryMapper::closeFile() {
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
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
void MemoryMapper::openFile() {
    // Ensure directory exists
    std::filesystem::path path(filename_);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    fd_ = open(filename_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ == -1) {
        throw std::runtime_error("Failed to open file: " + filename_ + " Error: " + strerror(errno));
    }

    // Resize file
    if (ftruncate(fd_, size_) == -1) {
        close(fd_);
        throw std::runtime_error("Failed to resize file. Error: " + std::string(strerror(errno)));
    }

    data_ = mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Failed to mmap file. Error: " + std::string(strerror(errno)));
    }
}

void MemoryMapper::closeFile() {
    if (data_) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}
#endif
