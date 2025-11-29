#pragma once

#include <string>
#include <cstdint>
#include <stdexcept>
#include "FileFormat.hpp"

class MemoryMapper {
public:
    // size is the size of the DATA section. The file will be size + 4096.
    MemoryMapper(const std::string& filename, size_t data_size, bool create_new = false);
    ~MemoryMapper();

    void* get_data();
    const void* get_data() const;
    
    pycauset::FileHeader* get_header();
    const pycauset::FileHeader* get_header() const;

    size_t get_data_size() const;

private:
    std::string filename_;
    size_t data_size_;
    void* mapped_ptr_; // Pointer to the start of the file (header)
    
#ifdef _WIN32
    void* hFile_;
    void* hMapping_;
#else
    int fd_;
#endif

    void open_file(bool create_new);
    void close_file();
};
