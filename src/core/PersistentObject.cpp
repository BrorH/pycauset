#include "pycauset/core/PersistentObject.hpp"
#include "pycauset/core/SystemUtils.hpp"
#include "pycauset/core/MemoryMapper.hpp"

#include <stdexcept>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <algorithm>

#include "pycauset/core/StorageUtils.hpp"

PersistentObject::PersistentObject() 
    : mapper_(nullptr), 
      matrix_type_(pycauset::MatrixType::UNKNOWN), 
      data_type_(pycauset::DataType::UNKNOWN), 
      rows_(0), cols_(0), seed_(0), scalar_(1.0), 
      is_transposed_(false), is_temporary_(false) {}

PersistentObject::PersistentObject(std::unique_ptr<MemoryMapper> mapper,
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
}

#include <iostream>

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
    if (backing_file.empty()) {
        // Check user-defined threshold first
        uint64_t threshold = pycauset::get_memory_threshold();
        bool force_disk = (final_size > threshold);

        if (!force_disk) {
            // Check if we should use RAM-backed storage
            // Use all available RAM logic (Step 0 of Hyper-Optimization Plan)
            uint64_t available_ram = pycauset::SystemUtils::get_available_ram();
            // Reserve 10% or 500MB, whichever is larger, for OS stability
            uint64_t reserve = std::max<uint64_t>(available_ram / 10, 500 * 1024 * 1024);

            if (final_size + reserve > available_ram) {
                force_disk = true;
            }
        }

        if (!force_disk) {
            path = ":memory:";
        } else {
            path = resolve_backing_file(backing_file, fallback_prefix);
        }
    } else {
        path = resolve_backing_file(backing_file, fallback_prefix);
    }
    
    // Create new file, no header space needed
    mapper_ = std::make_unique<MemoryMapper>(path, final_size, offset, create_new);

    // Initialize members
    matrix_type_ = matrix_type;
    data_type_ = data_type;
    rows_ = rows;
    cols_ = cols;
    scalar_ = 1.0;
    is_temporary_ = (backing_file.empty() || path == ":memory:" || backing_file.ends_with(".tmp"));
    is_transposed_ = false;
    seed_ = 0;
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
