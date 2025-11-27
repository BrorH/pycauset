#pragma once

#include <string>
#include <cstdint>
#include <stdexcept>

class MemoryMapper {
public:
    MemoryMapper(const std::string& filename, size_t size);
    ~MemoryMapper();

    void* getData();
    const void* getData() const;
    size_t getSize() const;

private:
    std::string filename_;
    size_t size_;
    void* data_;
    
#ifdef _WIN32
    void* hFile_;
    void* hMapping_;
#else
    int fd_;
#endif

    void openFile();
    void closeFile();
};
