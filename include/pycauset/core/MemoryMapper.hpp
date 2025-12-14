#pragma once

#include <string>
#include <cstdint>
#include <stdexcept>

class MemoryMapper {
public:
    // size is the size of the DATA section.
    MemoryMapper(const std::string& filename, size_t data_size, size_t offset = 0, bool create_new = false);
    ~MemoryMapper();

    void* get_data();
    const void* get_data() const;
    
    size_t get_data_size() const;
    size_t get_offset() const { return offset_; }
    const std::string& get_filename() const { return filename_; }

    void flush();
    void flush(void* ptr, size_t size);

    // Advanced mapping control for large files
    void unmap();
    void map_all();
    void* map_region(size_t offset, size_t size);
    void unmap_region(void* ptr);
    static size_t get_granularity();

    // Pinning support (VirtualLock/mlock)
    bool pin_region(void* ptr, size_t size) const;
    void unpin_region(void* ptr, size_t size) const;

private:
    std::string filename_;
    size_t data_size_;
    size_t offset_;
    void* mapped_ptr_; 
    void* base_ptr_; // The actual start of the mapping (aligned)
    
#ifdef _WIN32
    void* hFile_;
    void* hMapping_;
#else
    int fd_;
#endif
    bool is_pinned_ = false;

    void open_file(bool create_new);
    void close_file();
};
