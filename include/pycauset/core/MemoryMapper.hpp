#pragma once

#include <string>
#include <cstdint>
#include <stdexcept>

class MemoryMapper {
public:
    // size is the size of the DATA section.
    // private_writes: map an existing file copy-on-write (MAP_PRIVATE / FILE_MAP_COPY). Reads
    // come from the file; a written page becomes private memory; the file never changes. Used
    // for saved snapshots (pc.load, F-093). A private mapping cannot be remapped.
    MemoryMapper(const std::string& filename, size_t data_size, size_t offset = 0, bool create_new = false,
                 bool private_writes = false);
    ~MemoryMapper();

    void* get_data();
    const void* get_data() const;
    
    size_t get_data_size() const;
    size_t get_offset() const { return offset_; }
    const std::string& get_filename() const { return filename_; }
    bool is_private() const { return private_; }

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
    struct FileHeader {
        char magic[8];      // "PYCAUSET"
        uint32_t version;   // 1
        uint8_t reserved[52];
    };
    static_assert(sizeof(FileHeader) == 64, "FileHeader must be 64 bytes");

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
    bool private_ = false;
    bool heap_ = false;  // :memory: storage from the C heap (no mapping)

    void open_file(bool create_new);
    void close_file();
};
