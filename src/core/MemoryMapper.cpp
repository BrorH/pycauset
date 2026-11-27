#include <new>
#include <cstdlib>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "pycauset/core/MemoryMapper.hpp"

namespace {
thread_local int g_uninitialized_depth = 0;
}
pycauset::UninitializedStorage::UninitializedStorage() { ++g_uninitialized_depth; }
pycauset::UninitializedStorage::~UninitializedStorage() { --g_uninitialized_depth; }
bool pycauset::UninitializedStorage::active() { return g_uninitialized_depth > 0; }

#include "pycauset/compute/ComputeContext.hpp"
#include <filesystem>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
// Helper to enable privileges (SE_MANAGE_VOLUME_NAME for SetFileValidData)
static bool EnablePrivilege(LPCTSTR lpszPrivilege) {
    HANDLE hToken;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
        return false;

    TOKEN_PRIVILEGES tp;
    if (!LookupPrivilegeValue(NULL, lpszPrivilege, &tp.Privileges[0].Luid)) {
        CloseHandle(hToken);
        return false;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    if (!AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(TOKEN_PRIVILEGES), (PTOKEN_PRIVILEGES)NULL, (PDWORD)NULL)) {
        CloseHandle(hToken);
        return false;
    }

    if (GetLastError() == ERROR_NOT_ALL_ASSIGNED) {
        CloseHandle(hToken);
        return false;
    }

    CloseHandle(hToken);
    return true;
}
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <atomic>

#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif
#endif

MemoryMapper::MemoryMapper(const std::string& filename, size_t data_size, size_t offset, bool create_new,
                           bool private_writes)
    : filename_(filename), data_size_(data_size), offset_(offset), mapped_ptr_(nullptr), base_ptr_(nullptr),
#ifdef _WIN32
      hFile_(INVALID_HANDLE_VALUE), hMapping_(nullptr),
#else
      fd_(-1),
#endif
      is_pinned_(false), private_(private_writes && !create_new && filename != ":memory:") {
    open_file(create_new);
}

MemoryMapper::~MemoryMapper() {
    close_file();
}

void* MemoryMapper::get_data() {
    return mapped_ptr_;
}

const void* MemoryMapper::get_data() const {
    return mapped_ptr_;
}

size_t MemoryMapper::get_data_size() const {
    return data_size_;
}

size_t MemoryMapper::get_granularity() {
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return sysInfo.dwAllocationGranularity;
#else
    return sysconf(_SC_PAGE_SIZE);
#endif
}

#ifdef _WIN32
void MemoryMapper::open_file(bool create_new) {
    if (filename_ == ":memory:") {
        // NOTE: We intentionally do NOT auto-pin ":memory:" backing via
        // cudaHostAlloc when the GPU is active. Permanent pinning of every
        // allocation slows CPU-side elementwise/GEMM access (page-locked memory
        // + the costlier cudaHostAlloc path) and pushed add/multiply/matmul
        // below NumPy parity once the GPU was active. GPU transfers pin the host
        // buffers they actually move on-demand (ScopedPinner) instead, so CPU
        // workloads stay on plain VirtualAlloc while GPU transfers still get
        // fast DMA where it matters.
        hFile_ = INVALID_HANDLE_VALUE;
        offset_ = 0; // Ignore offset for memory-only
    } else {
        // Ensure directory exists
        // Use char8_t cast for UTF-8 string
        std::filesystem::path path(reinterpret_cast<const char8_t*>(filename_.c_str()));
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }

        DWORD creationDisposition = create_new ? CREATE_ALWAYS : OPEN_EXISTING;

        hFile_ = CreateFileW(
            path.c_str(),
            private_ ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE),
            FILE_SHARE_READ | FILE_SHARE_WRITE, 
            NULL,
            creationDisposition,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );

        if (hFile_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file: " + filename_);
        }

        // R1_SAFETY: File Header Logic
        FileHeader header;
        DWORD bytesProcessed = 0;
        
        if (create_new) {
            std::memcpy(header.magic, "PYCAUSET", 8);
            header.version = 1;
            std::memset(header.reserved, 0, sizeof(header.reserved));
            
            if (!WriteFile(hFile_, &header, sizeof(header), &bytesProcessed, NULL) || bytesProcessed != sizeof(header)) {
                CloseHandle(hFile_);
                throw std::runtime_error("Failed to write file header");
            }
            offset_ += sizeof(FileHeader);
        } else {
            if (!ReadFile(hFile_, &header, sizeof(header), &bytesProcessed, NULL) || bytesProcessed != sizeof(header)) {
                CloseHandle(hFile_);
                throw std::runtime_error("Failed to read file header or file too short");
            }
            if (std::memcmp(header.magic, "PYCAUSET", 8) != 0) {
                CloseHandle(hFile_);
                throw std::runtime_error("Invalid file format (bad magic): " + filename_);
            }
            if (header.version != 1) {
                CloseHandle(hFile_);
                throw std::runtime_error("Unsupported file version: " + std::to_string(header.version));
            }
            
            // Check if it's a Simple Header (all reserved bytes are 0)
            // But only if offset is 0! If caller provided an offset (e.g. 4096 for a snapshot),
            // we trust them and do not skip another header.
            bool is_simple_header = true;
            for (int i = 0; i < 52; ++i) {
                if (header.reserved[i] != 0) {
                    is_simple_header = false;
                    break;
                }
            }
            
            if (is_simple_header && offset_ == 0) {
                offset_ += sizeof(FileHeader);
            } else if (offset_ == 0) {
                // A container header opened without its payload offset (see the POSIX branch).
                CloseHandle(hFile_);
                throw std::runtime_error("Container file opened without a payload offset: " + filename_);
            }
        }
    }

    size_t granularity = get_granularity();
    size_t aligned_offset = (offset_ / granularity) * granularity;
    size_t adjustment = offset_ - aligned_offset;
    size_t map_size = data_size_ + adjustment;

    size_t total_required_size = offset_ + data_size_;
    LARGE_INTEGER liSize;
    liSize.QuadPart = total_required_size;

    if (create_new) {
        if (hFile_ != INVALID_HANDLE_VALUE) {
            if (!SetFilePointerEx(hFile_, liSize, NULL, FILE_BEGIN)) {
                CloseHandle(hFile_);
                throw std::runtime_error("Failed to set file pointer");
            }
            if (!SetEndOfFile(hFile_)) {
                CloseHandle(hFile_);
                throw std::runtime_error("Failed to set end of file");
            }

            // Optimization: SetFileValidData to skip zero-filling
            // This requires SE_MANAGE_VOLUME_NAME privilege.
            // If it fails, we just proceed (OS will zero-fill on write/fault).
            if (EnablePrivilege(SE_MANAGE_VOLUME_NAME)) {
                // On failure Windows zero-fills on first write: slower, still correct.
                (void)SetFileValidData(hFile_, liSize.QuadPart);
            }
        }
    } else {
        if (hFile_ == INVALID_HANDLE_VALUE) {
             if (data_size_ == 0) {
                 throw std::runtime_error("Cannot open existing anonymous mapping without size");
             }
             // For :memory:, total_required_size is just data_size_ (offset is 0)
             liSize.QuadPart = data_size_;
        } else {
            LARGE_INTEGER fileSize;
            if (!GetFileSizeEx(hFile_, &fileSize)) {
                CloseHandle(hFile_);
                throw std::runtime_error("Failed to get file size");
            }
            
            if (data_size_ == 0) {
                // Map from offset to end of file
                if (static_cast<size_t>(fileSize.QuadPart) <= offset_) {
                     CloseHandle(hFile_);
                     throw std::runtime_error("File is too small for offset");
                }
                data_size_ = static_cast<size_t>(fileSize.QuadPart) - offset_;
                // Recalculate map_size if data_size_ changed
                map_size = data_size_ + adjustment;
                liSize.QuadPart = fileSize.QuadPart;
            } else {
                if (static_cast<size_t>(fileSize.QuadPart) < total_required_size) {
                     CloseHandle(hFile_);
                     throw std::runtime_error("File is smaller than expected size");
                }
            }
        }
    }

    if (data_size_ == 0) {
        // Zero-sized mapping requested (e.g. IdentityMatrix).
        // No need to map anything.
        return;
    }

    // Anonymous (:memory:) storage: allocate with VirtualAlloc instead of a
    // pagefile-backed section (CreateFileMappingA(INVALID_HANDLE_VALUE)). The
    // pagefile-backed path commits a page from the paging file on every first
    // write, which costs several ms for an 8MB result and dominates small
    // elementwise ops (add/multiply). VirtualAlloc gives fast anonymous commit
    // (the same class as malloc), closing most of the gap vs NumPy.
    if (hFile_ == INVALID_HANDLE_VALUE) {
        // RAM objects from the C heap (see the POSIX branch).
        base_ptr_ = pycauset::UninitializedStorage::active() ? std::malloc(map_size) : std::calloc(1, map_size);
        if (base_ptr_ == nullptr) throw std::bad_alloc();
        heap_ = true;
        hMapping_ = NULL;
        mapped_ptr_ = base_ptr_;
        return;
    }
    if (hFile_ == INVALID_HANDLE_VALUE) {
        base_ptr_ = VirtualAlloc(NULL, map_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (base_ptr_ == NULL) {
            throw std::runtime_error("Failed to allocate anonymous memory (VirtualAlloc)");
        }
        hMapping_ = NULL;
        mapped_ptr_ = static_cast<char*>(base_ptr_) + adjustment;
        return;
    }

    hMapping_ = CreateFileMappingA(
        hFile_,
        NULL,
        private_ ? PAGE_WRITECOPY : PAGE_READWRITE,
        liSize.HighPart,
        liSize.LowPart,
        NULL
    );

    if (hMapping_ == NULL) {
        if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
        throw std::runtime_error("Failed to create file mapping");
    }

    ULARGE_INTEGER liOffset;
    liOffset.QuadPart = aligned_offset;

    base_ptr_ = MapViewOfFile(
        hMapping_,
        private_ ? FILE_MAP_COPY : FILE_MAP_ALL_ACCESS,
        liOffset.HighPart,
        liOffset.LowPart,
        map_size
    );

    if (base_ptr_ == NULL) {
        CloseHandle(hMapping_);
        if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
        throw std::runtime_error("Failed to map view of file");
    }
    
    mapped_ptr_ = static_cast<char*>(base_ptr_) + adjustment;
    
}

void MemoryMapper::close_file() {
    if (heap_) {
        std::free(base_ptr_);
        base_ptr_ = nullptr;
        mapped_ptr_ = nullptr;
        heap_ = false;
        return;
    }
    if (base_ptr_) {
        // Check if it was pinned memory
        if (is_pinned_) {
            pycauset::ComputeContext::instance().free_pinned(base_ptr_);
        } else if (hMapping_ == NULL) {
            // Anonymous (:memory:) storage allocated with VirtualAlloc.
            VirtualFree(base_ptr_, 0, MEM_RELEASE);
        } else {
            UnmapViewOfFile(base_ptr_);
        }
        base_ptr_ = nullptr;
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
void MemoryMapper::open_file(bool create_new) {
    if (filename_ == ":memory:") {
        // NOTE: no auto-pin via cudaHostAlloc here, see the Windows branch
        // comment. GPU transfers pin on-demand (ScopedPinner) instead.

        // RAM-backed storage is plain private anonymous memory (what malloc uses): no file
        // descriptor, pages faulted lazily on first write, transparent huge pages allowed.
        // The former memfd/shm + MAP_SHARED + MAP_POPULATE path zeroed and faulted every page
        // up front, and shared-memory pages fault slower: allocating a 3000x3000 float64
        // result took 23 ms (NumPy ~8 ms), dominating element-wise ops (F-012). unmap() /
        // map_region() treat :memory: like the Windows VirtualAlloc path (no remapping).
        fd_ = -1;
        offset_ = 0;
    } else {
        std::filesystem::path path(filename_);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }

        int flags = private_ ? O_RDONLY : O_RDWR;
        if (create_new) flags |= O_CREAT | O_TRUNC;
        
        fd_ = open(filename_.c_str(), flags, 0644);
        if (fd_ == -1) {
            throw std::runtime_error("Failed to open file: " + filename_ + " (" + std::strerror(errno) + ")");
        }
        FileHeader header;
        if (create_new) {


            std::memcpy(header.magic, "PYCAUSET", 8);
            header.version = 1;
            std::memset(header.reserved, 0, sizeof(header.reserved));
            
            if (write(fd_, &header, sizeof(header)) != sizeof(header)) {
                close(fd_);
                throw std::runtime_error("Failed to write file header");
            }
            offset_ += sizeof(FileHeader);
            
            // R1_SAFETY: On Linux, we must extend the file to the full size before mmap.
            // Unlike Windows CreateFileMapping which extends automatically.
            if (ftruncate(fd_, offset_ + data_size_) == -1) {
                close(fd_);
                throw std::runtime_error("Failed to resize file: " + std::string(std::strerror(errno)));
            }
        } else {
            if (read(fd_, &header, sizeof(header)) != sizeof(header)) {
                close(fd_);
                throw std::runtime_error("Failed to read file header or file too short");
            }
            if (std::memcmp(header.magic, "PYCAUSET", 8) != 0) {
                close(fd_);
                throw std::runtime_error("Invalid file format (bad magic): " + filename_);
            }
            if (header.version != 1) {
                close(fd_);
                throw std::runtime_error("Unsupported file version: " + std::to_string(header.version));
            }
            
            bool is_simple_header = true;
            for (int i = 0; i < 52; ++i) {
                if (header.reserved[i] != 0) {
                    is_simple_header = false;
                    break;
                }
            }
            
            if (is_simple_header && offset_ == 0) {
                offset_ += sizeof(FileHeader);
            } else if (offset_ == 0) {
                // A container header (non-zero fields after the version) opened without its
                // payload offset: mapping from byte 0 would read the header as data.
                close(fd_);
                throw std::runtime_error("Container file opened without a payload offset: " + filename_);
            }
        }
    }

    size_t granularity = get_granularity();
    size_t aligned_offset = (offset_ / granularity) * granularity;
    size_t adjustment = offset_ - aligned_offset;
    size_t map_size = data_size_ + adjustment;

    size_t total_required_size = offset_ + data_size_;

    // File sizing/validation only applies to real files. For :memory:, the shm/memfd
    // in the :memory: branch above is already sized to data_size_ and has no header,
    // so neither the create_new resize nor the load-time fstat checks should run.
    if (filename_ != ":memory:") {
        if (create_new) {
            if (fd_ != -1) {
                // Try fallocate first to pre-allocate blocks (avoids fragmentation and some metadata updates)
                // 0 = default mode (allocate and initialize to zero)
                // We could use FALLOC_FL_KEEP_SIZE if we wanted, but we want to set size.
                // Note: fallocate is not standard POSIX, but available on Linux.
                // If it fails (e.g. not supported by FS), fallback to ftruncate.
#ifdef __linux__
                if (fallocate(fd_, 0, 0, total_required_size) != 0) {
                    if (ftruncate(fd_, total_required_size) == -1) {
                        close(fd_);
                        throw std::runtime_error("Failed to resize file (fallocate & ftruncate failed)");
                    }
                }
#else
                if (ftruncate(fd_, total_required_size) == -1) {
                    close(fd_);
                    throw std::runtime_error("Failed to resize file");
                }
#endif
            }
        } else {
            if (fd_ != -1) {
                struct stat st;
                if (fstat(fd_, &st) == -1) {
                    close(fd_);
                    throw std::runtime_error("Failed to stat file");
                }
                
                if (data_size_ == 0) {
                    if (static_cast<size_t>(st.st_size) <= offset_) {
                        close(fd_);
                        throw std::runtime_error("File too small for offset");
                    }
                    data_size_ = static_cast<size_t>(st.st_size) - offset_;
                    // Recalculate map_size
                    map_size = data_size_ + adjustment;
                } else {
                    if (static_cast<size_t>(st.st_size) < total_required_size) {
                        close(fd_);
                        throw std::runtime_error("File smaller than expected");
                    }
                }
            } else {
                 if (data_size_ == 0) throw std::runtime_error("Anonymous mapping requires size");
            }
        }
    }

    const bool anonymous = (filename_ == ":memory:");
    int map_flags = anonymous ? (MAP_PRIVATE | MAP_ANONYMOUS) : (private_ ? MAP_PRIVATE : MAP_SHARED);

    // R1_SAFETY: mmap with length 0 is invalid (EINVAL).
    if (map_size == 0) {
        return;
    }

    // RAM objects come from the C heap (calloc: zeroed, 16-byte aligned). Small ones avoid an
    // mmap + munmap per result (~20 us for a 4 x 4, F-096); large ones are reused by the
    // allocator after a free instead of page-faulting a fresh mapping on every result (an
    // 8 MB element-wise result spent ~0.8 ms in page faults, more than the arithmetic). calloc
    // skips the memset for memory it gets fresh from the kernel.
    if (anonymous) {
        base_ptr_ = pycauset::UninitializedStorage::active() ? std::malloc(map_size) : std::calloc(1, map_size);
        if (base_ptr_ == nullptr) throw std::bad_alloc();
        heap_ = true;
        mapped_ptr_ = base_ptr_;
        return;
    }

#ifdef __linux__
    // New files are about to be written: pre-fault them. Anonymous memory is not
    // pre-faulted (first touch by the writing kernel is cheaper than populate + rewrite).
    if (create_new && !anonymous) map_flags |= MAP_POPULATE;
#endif

    base_ptr_ = mmap(NULL, map_size, PROT_READ | PROT_WRITE, map_flags, anonymous ? -1 : fd_, aligned_offset);
#ifdef __linux__
    // No MADV_HUGEPAGE: with the common `defrag = madvise` kernel setting it makes page faults
    // compact memory synchronously and khugepaged collapse the region later; a freshly imported
    // 8 MB matrix then read 50x slower for its first calls (norm 32 ms vs 0.6 ms, F-096).
#endif
    
    if (base_ptr_ == MAP_FAILED) {
        if (fd_ != -1) close(fd_);
        throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
    }
    
    mapped_ptr_ = static_cast<char*>(base_ptr_) + adjustment;
}

void MemoryMapper::close_file() {
    if (heap_) {
        std::free(base_ptr_);
        base_ptr_ = nullptr;
        mapped_ptr_ = nullptr;
        heap_ = false;
        return;
    }
    if (base_ptr_ && base_ptr_ != MAP_FAILED) {
        if (is_pinned_) {
            pycauset::ComputeContext::instance().free_pinned(base_ptr_);
        } else {
            size_t granularity = get_granularity();
            size_t aligned_offset = (offset_ / granularity) * granularity;
            size_t adjustment = offset_ - aligned_offset;
            size_t map_size = data_size_ + adjustment;
            
            munmap(base_ptr_, map_size);
        }
        base_ptr_ = nullptr;
        mapped_ptr_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}
#endif

void MemoryMapper::flush() {
    if (mapped_ptr_) {
#ifdef _WIN32
        if (hMapping_ != NULL) {
            FlushViewOfFile(mapped_ptr_, 0);
        }
        if (hFile_ != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(hFile_);
        }
#else
        msync(mapped_ptr_, data_size_, MS_SYNC);
#endif
    }
}

void MemoryMapper::flush(void* ptr, size_t size) {
    if (ptr && size > 0) {
#ifdef _WIN32
        if (hMapping_ != NULL) {
            FlushViewOfFile(ptr, size);
        }
        // R1_SAFETY: Ensure metadata/file size changes are also committed
        if (hFile_ != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(hFile_);
        }
#else
        msync(ptr, size, MS_SYNC);
#endif
    }
}

void MemoryMapper::unmap() {
    if (private_) throw std::logic_error("MemoryMapper: a copy-on-write (snapshot) mapping cannot be remapped");
#ifdef _WIN32
    if (filename_ == ":memory:") {
        // VirtualAlloc memory is a single committed region, not a mapped file
        // view; there is nothing to unmap (and UnmapViewOfFile would be wrong
        // for it). fill_random() calls this only to shed address space for large
        // disk-backed files, so for RAM-backed storage it is a no-op.
        return;
    }
    if (mapped_ptr_) {
        UnmapViewOfFile(mapped_ptr_);
        mapped_ptr_ = nullptr;
    }
#else
    if (filename_ == ":memory:") return;  // anonymous memory: nothing to remap (see open_file)
    if (is_pinned_) return; // Pinned memory cannot be unmapped/remapped dynamically here

    if (base_ptr_ && base_ptr_ != MAP_FAILED) {
        size_t granularity = get_granularity();
        size_t aligned_offset = (offset_ / granularity) * granularity;
        size_t adjustment = offset_ - aligned_offset;
        size_t map_size = data_size_ + adjustment;

        munmap(base_ptr_, map_size);
        mapped_ptr_ = nullptr;
        base_ptr_ = nullptr;
    }
#endif
}


void MemoryMapper::map_all() {
    if (private_) throw std::logic_error("MemoryMapper: a copy-on-write (snapshot) mapping cannot be remapped");
    if (mapped_ptr_) return; // Already mapped
#ifdef _WIN32
    if (filename_ == ":memory:") {
        // Restore the (never-freed) VirtualAlloc pointer.
        mapped_ptr_ = base_ptr_;
        return;
    }
    if (!hMapping_) throw std::runtime_error("File mapping handle is invalid");
    
    ULARGE_INTEGER liOffset;
    liOffset.QuadPart = offset_;

    mapped_ptr_ = MapViewOfFile(hMapping_, FILE_MAP_ALL_ACCESS, liOffset.HighPart, liOffset.LowPart, data_size_);
    if (!mapped_ptr_) {
        throw std::runtime_error("Failed to map view of file");
    }
#else
    if (fd_ == -1 && filename_ != ":memory:") throw std::runtime_error("File descriptor invalid");
    
    int map_flags = MAP_SHARED;
    if (fd_ == -1) map_flags |= MAP_ANONYMOUS;

    // R1_SAFETY: mmap offset must be page-aligned
    size_t granularity = get_granularity();
    size_t aligned_offset = (offset_ / granularity) * granularity;
    size_t adjustment = offset_ - aligned_offset;
    size_t map_size = data_size_ + adjustment;

    void* base = mmap(NULL, map_size, PROT_READ | PROT_WRITE, map_flags, fd_, aligned_offset);
    if (base == MAP_FAILED) {
        throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
    }
    
    base_ptr_ = base;
    mapped_ptr_ = static_cast<char*>(base) + adjustment;
#endif
}

void* MemoryMapper::map_region(size_t offset, size_t size) {
    if (private_) throw std::logic_error("MemoryMapper: a copy-on-write (snapshot) mapping cannot be remapped");
#ifdef _WIN32
    if (filename_ == ":memory:") {
        // The whole VirtualAlloc region is already committed and addressable;
        // return a pointer into it. unmap_region() is a matching no-op.
        return static_cast<char*>(base_ptr_) + offset;
    }
    if (!hMapping_) throw std::runtime_error("File mapping handle is invalid");
    
    ULARGE_INTEGER liOffset;
    liOffset.QuadPart = offset;
    
    void* ptr = MapViewOfFile(hMapping_, FILE_MAP_ALL_ACCESS, liOffset.HighPart, liOffset.LowPart, size);
    if (!ptr) {
        throw std::runtime_error("Failed to map region");
    }
    return ptr;
#else
    if (filename_ == ":memory:") return static_cast<char*>(base_ptr_) + offset;  // interior pointer
    if (fd_ == -1) throw std::runtime_error("File descriptor invalid");

    // R1_SAFETY: mmap offset must be page-aligned
    size_t granularity = get_granularity();
    size_t aligned_offset = (offset / granularity) * granularity;
    size_t adjustment = offset - aligned_offset;
    size_t map_size = size + adjustment;

    void* ptr = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, aligned_offset);
    if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
    
    return static_cast<char*>(ptr) + adjustment;
#endif
}

void MemoryMapper::unmap_region(void* ptr) {
#ifdef _WIN32
    if (filename_ == ":memory:") return; // map_region returned an interior pointer; nothing to free
    UnmapViewOfFile(ptr);
#else
    // munmap requires size, which we don't have here easily without tracking.
    // This API might need revision for POSIX if partial unmapping is common.
    // For now, assuming the user manages this or we don't support partial unmap without size.
    // But since this is a refactor of existing code, I'll leave it as a placeholder or assume full unmap isn't called this way.
#endif
}

bool MemoryMapper::pin_region(void* ptr, size_t size) const {
#ifdef _WIN32
    if (VirtualLock(ptr, size)) {
        return true;
    }
    return false;
#else
    if (mlock(ptr, size) == 0) {
        return true;
    }
    return false;
#endif
}

void MemoryMapper::unpin_region(void* ptr, size_t size) const {
#ifdef _WIN32
    VirtualUnlock(ptr, size);
#else
    munlock(ptr, size);
#endif
}
