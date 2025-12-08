#include "pycauset/core/PersistentObject.hpp"
#include "pycauset/core/SystemUtils.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/IOAccelerator.hpp"

#include <stdexcept>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>

#include "pycauset/core/StorageUtils.hpp"

PersistentObject::PersistentObject() 
    : mapper_(nullptr), 
      matrix_type_(pycauset::MatrixType::UNKNOWN), 
      data_type_(pycauset::DataType::UNKNOWN), 
      rows_(0), cols_(0), seed_(0), scalar_(1.0), 
      is_transposed_(false), is_temporary_(false),
      storage_state_(pycauset::core::StorageState::DISK_BACKED) {}

PersistentObject::PersistentObject(std::shared_ptr<MemoryMapper> mapper,
                                   pycauset::MatrixType matrix_type,
                                   pycauset::DataType data_type,
                                   uint64_t rows,
                                   uint64_t cols,
                                   uint64_t seed,
                                   double scalar,
                                   bool is_transposed,
                                   bool is_temporary)
    : mapper_(std::move(mapper)),
      matrix_type_(matrix_type),
      data_type_(data_type),
      rows_(rows),
      cols_(cols),
      seed_(seed),
      scalar_(scalar),
      is_transposed_(is_transposed),
      is_temporary_(is_temporary)
{
    // Determine initial state
    if (mapper_ && mapper_->get_filename() == ":memory:") {
        storage_state_ = pycauset::core::StorageState::RAM_ONLY;
        pycauset::core::MemoryGovernor::instance().register_object(this, mapper_->get_data_size());
    } else {
        storage_state_ = pycauset::core::StorageState::DISK_BACKED;
    }
}

PersistentObject::~PersistentObject() {
    close();
}

std::string PersistentObject::get_backing_file() const {
    if (mapper_) {
        return mapper_->get_filename();
    }
    return "";
}

void PersistentObject::close() {
    if (mapper_) {
        // Unregister from Governor if we were in RAM
        if (storage_state_ == pycauset::core::StorageState::RAM_ONLY) {
            pycauset::core::MemoryGovernor::instance().unregister_object(this);
        }

        bool temp = is_temporary_;
        std::string path = mapper_->get_filename();
        // std::cout << "Closing object. Temp: " << temp << ", Path: " << path << std::endl;
        mapper_.reset(); // Close file mapping

        if (temp && !path.empty() && path != ":memory:") {
            try {
                // Use char8_t cast for UTF-8 string to handle Unicode paths
                std::filesystem::path p(reinterpret_cast<const char8_t*>(path.c_str()));
                if (std::filesystem::remove(p)) {
                    // std::cout << "Deleted temp file: " << path << std::endl;
                } else {
                    // std::cerr << "Failed to delete temp file: " << path << std::endl;
                }
            } catch (const std::exception& e) {
                // std::cerr << "Exception deleting temp file: " << path << " - " << e.what() << std::endl;
            } catch (...) {
                // std::cerr << "Unknown exception deleting temp file: " << path << std::endl;
            }
        }
    }
}

void PersistentObject::ensure_unique() {
    if (mapper_ && mapper_.use_count() > 1) {
        size_t size = mapper_->get_data_size();
        std::string new_path;
        bool use_ram = (storage_state_ == pycauset::core::StorageState::RAM_ONLY);
        
        if (use_ram) {
             if (!pycauset::core::MemoryGovernor::instance().request_ram(size)) {
                 use_ram = false;
             }
        }
        
        if (use_ram) {
            new_path = ":memory:";
        } else {
            new_path = pycauset::make_unique_storage_file("cow_copy");
        }
        
        auto new_mapper = std::make_unique<MemoryMapper>(new_path, size, 0, true);
        std::memcpy(new_mapper->get_data(), mapper_->get_data(), size);
        
        if (!use_ram) {
            new_mapper->flush();
        }
        
        mapper_ = std::move(new_mapper);
        accelerator_.reset();
        
        if (!use_ram) {
            storage_state_ = pycauset::core::StorageState::DISK_BACKED;
        } else {
            storage_state_ = pycauset::core::StorageState::RAM_ONLY;
        }
    }
}

void PersistentObject::touch() {
    if (storage_state_ == pycauset::core::StorageState::RAM_ONLY) {
        pycauset::core::MemoryGovernor::instance().touch(this);
    }
}

bool PersistentObject::spill_to_disk() {
    if (storage_state_ != pycauset::core::StorageState::RAM_ONLY || !mapper_) {
        return false;
    }

    // 1. Create temp file path
    std::string temp_path = pycauset::make_unique_storage_file("spill");
    size_t size = mapper_->get_data_size();

    // Log warning for large files
    if (size > 1024 * 1024 * 100) { // 100MB
        std::cout << "PyCauset: Evicting object to disk to free RAM (" << (size / 1024 / 1024) << " MB)..." << std::endl;
    }

    try {
        storage_state_ = pycauset::core::StorageState::TRANSITIONING;

        // 2. Create new file-backed mapper
        auto new_mapper = std::make_unique<MemoryMapper>(temp_path, size, 0, true);

        // 3. Copy data
        std::memcpy(new_mapper->get_data(), mapper_->get_data(), size);
        new_mapper->flush();

        // 4. Swap mappers
        mapper_ = std::move(new_mapper);
        accelerator_.reset();

        // 5. Update state
        storage_state_ = pycauset::core::StorageState::DISK_BACKED;
        is_temporary_ = true; // Spilled files are always temporary

        // 6. Unregister from Governor (RAM usage is now 0)
        pycauset::core::MemoryGovernor::instance().unregister_object(this);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to spill object to disk: " << e.what() << std::endl;
        // Revert state (data is still in RAM)
        storage_state_ = pycauset::core::StorageState::RAM_ONLY;
        return false;
    }
}

bool PersistentObject::promote_to_ram() {
    if (storage_state_ != pycauset::core::StorageState::DISK_BACKED || !mapper_) {
        return false;
    }

    size_t size = mapper_->get_data_size();

    // 1. Ask Governor for RAM
    if (!pycauset::core::MemoryGovernor::instance().request_ram(size)) {
        return false;
    }

    try {
        storage_state_ = pycauset::core::StorageState::TRANSITIONING;

        // 2. Create new RAM-backed mapper
        auto new_mapper = std::make_unique<MemoryMapper>(":memory:", size, 0, true);

        // 3. Copy data
        std::memcpy(new_mapper->get_data(), mapper_->get_data(), size);

        // 4. Swap mappers
        // Note: Old file is closed when unique_ptr is overwritten.
        // If it was temporary, it will be deleted by close() logic if we didn't move it out.
        // Wait, mapper_ is overwritten. The old mapper's destructor runs.
        // If is_temporary_ is true, we want the old file deleted.
        // But PersistentObject::close() handles deletion based on is_temporary_.
        // MemoryMapper destructor just unmaps. It doesn't delete files.
        // So we need to manually delete the old file if it was temporary.
        
        std::string old_path = mapper_->get_filename();
        bool was_temp = is_temporary_;

        mapper_ = std::move(new_mapper);
        accelerator_.reset();

        if (was_temp && old_path != ":memory:") {
             std::filesystem::remove(reinterpret_cast<const char8_t*>(old_path.c_str()));
        }

        // 5. Update state
        storage_state_ = pycauset::core::StorageState::RAM_ONLY;
        
        // 6. Register with Governor
        pycauset::core::MemoryGovernor::instance().register_object(this, size);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to promote object to RAM: " << e.what() << std::endl;
        storage_state_ = pycauset::core::StorageState::DISK_BACKED;
        // Also unregister if we failed after registering? No, we register at end.
        return false;
    }
}

void PersistentObject::initialize_storage(uint64_t size_in_bytes,
                                   const std::string& backing_file,
                                   const std::string& fallback_prefix,
                                   uint64_t min_size_bytes,
                                   pycauset::MatrixType matrix_type,
                                   pycauset::DataType data_type,
                                   uint64_t rows,
                                   uint64_t cols,
                                   size_t offset,
                                   bool create_new) {
    uint64_t final_size = size_in_bytes;
    if (final_size < min_size_bytes) {
        final_size = min_size_bytes;
    }
    
    std::string path;
    bool use_ram = false;

    if (backing_file.empty() || backing_file == ":memory:") {
        // Ask MemoryGovernor
        if (pycauset::core::MemoryGovernor::instance().request_ram(final_size)) {
            use_ram = true;
            path = ":memory:";
        } else {
            // Fallback to disk if RAM is full
            if (final_size > 1024 * 1024 * 100) { // Log for >100MB
                 std::cout << "PyCauset: RAM full (or object too large), falling back to disk for " 
                           << (final_size / 1024 / 1024) << " MB object." << std::endl;
            }
            path = resolve_backing_file("", fallback_prefix);
        }
    } else {
        path = resolve_backing_file(backing_file, fallback_prefix);
    }
    
    // Create new file, no header space needed
    mapper_ = std::make_unique<MemoryMapper>(path, final_size, offset, create_new);
    accelerator_.reset();

    // Initialize members
    matrix_type_ = matrix_type;
    data_type_ = data_type;
    rows_ = rows;
    cols_ = cols;
    scalar_ = 1.0;
    is_temporary_ = (backing_file.empty() || path == ":memory:" || backing_file.ends_with(".tmp"));
    is_transposed_ = false;
    seed_ = 0;

    if (use_ram) {
        storage_state_ = pycauset::core::StorageState::RAM_ONLY;
        pycauset::core::MemoryGovernor::instance().register_object(this, final_size);
    } else {
        storage_state_ = pycauset::core::StorageState::DISK_BACKED;
    }
}


std::string PersistentObject::copy_storage(const std::string& result_file_hint) const {
    std::string source_path = get_backing_file();
    
    if (mapper_) {
        mapper_->flush();
    }

    std::string dest_path = resolve_backing_file(result_file_hint, "copy");
    
    // Always use manual copy if we have a mapper.
    // This ensures we copy exactly what is in memory, avoiding issues with
    // filesystem cache synchronization or file locking on Windows.
    bool manual_copy = (mapper_ != nullptr);

    if (manual_copy) {
        // For RAM-backed objects or objects inside a container (offset > 0),
        // we must write the memory buffer to disk manually to extract just the data.
        // Use std::filesystem::path to handle Unicode paths correctly on Windows
        // Since we are using C++20 and pybind11 passes UTF-8 strings, we cast to char8_t
        std::filesystem::path dest(reinterpret_cast<const char8_t*>(dest_path.c_str()));
        std::ofstream outfile(dest, std::ios::binary);
        if (!outfile) {
            throw std::runtime_error("Failed to open destination file for writing: " + dest_path);
        }
        
        const char* data = static_cast<const char*>(mapper_->get_data());
        size_t size = mapper_->get_data_size();
        
        // std::cout << "Copying storage: " << size << " bytes from " << (void*)data << " to " << dest_path << std::endl;

        if (!outfile.write(data, size)) {
            throw std::runtime_error("Failed to write object data to disk: " + dest_path);
        }
        outfile.close();
    } else {
        if (source_path.empty()) {
            throw std::runtime_error("Cannot copy object without backing file");
        }
        // Use filesystem copy
        try {
            std::filesystem::path src(reinterpret_cast<const char8_t*>(source_path.c_str()));
            std::filesystem::path dst(reinterpret_cast<const char8_t*>(dest_path.c_str()));
            std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
        } catch (const std::filesystem::filesystem_error& e) {
            throw std::runtime_error("Failed to copy backing file: " + std::string(e.what()));
        }
    }
    
    return dest_path;
}

void PersistentObject::set_scalar(double s) {
    scalar_ = s;
}

void PersistentObject::set_seed(uint64_t seed) {
    seed_ = seed;
}

void PersistentObject::set_temporary(bool temp) {
    is_temporary_ = temp;
}

void PersistentObject::set_transposed(bool transposed) {
    is_transposed_ = transposed;
}

MemoryMapper* PersistentObject::require_mapper() {
    if (!mapper_) {
        throw std::runtime_error("Backing file has been closed.");
    }
    return mapper_.get();
}

const MemoryMapper* PersistentObject::require_mapper() const {
    if (!mapper_) {
        throw std::runtime_error("Backing file has been closed.");
    }
    return mapper_.get();
}

std::string PersistentObject::resolve_backing_file(const std::string& requested,
                                           const std::string& fallback_prefix) {
    if (!requested.empty()) {
        return requested;
    }
    return pycauset::make_unique_storage_file(fallback_prefix);
}

pycauset::core::IOAccelerator* PersistentObject::get_accelerator() const {
    if (!accelerator_) {
        if (mapper_) {
            accelerator_ = std::make_unique<pycauset::core::IOAccelerator>(mapper_.get());
        }
    }
    return accelerator_.get();
}

void PersistentObject::hint(const pycauset::core::MemoryHint& hint) const {
    if (auto* acc = get_accelerator()) {
        acc->process_hint(hint);
    }
}

