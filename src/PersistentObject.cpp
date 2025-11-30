#include "PersistentObject.hpp"

#include <stdexcept>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "StoragePaths.hpp"

PersistentObject::PersistentObject() : scalar_(1.0) {}

PersistentObject::PersistentObject(std::unique_ptr<MemoryMapper> mapper) 
    : mapper_(std::move(mapper)) {
    if (mapper_) {
        auto* header = mapper_->get_header();
        if (header->version >= 2) {
            scalar_ = header->scalar;
        } else {
            scalar_ = 1.0;
        }
    } else {
        scalar_ = 1.0;
    }
}

PersistentObject::~PersistentObject() {
    if (mapper_) {
        bool temp = false;
        try {
            temp = is_temporary();
        } catch (...) {}

        std::string path = mapper_->get_filename();
        mapper_.reset(); // Close file mapping

        if (temp && !path.empty() && path != ":memory:") {
            try {
                std::filesystem::remove(path);
            } catch (...) {
                // Best effort deletion
            }
        }
    }
}

std::string PersistentObject::get_backing_file() const {
    if (mapper_) {
        return mapper_->get_filename();
    }
    return "";
}

pycauset::DataType PersistentObject::get_data_type() const {
    return require_mapper()->get_header()->data_type;
}

pycauset::MatrixType PersistentObject::get_matrix_type() const {
    return require_mapper()->get_header()->matrix_type;
}

bool PersistentObject::is_transposed() const {
    return require_mapper()->get_header()->is_transposed != 0;
}

void PersistentObject::set_transposed(bool transposed) {
    require_mapper()->get_header()->is_transposed = transposed ? 1 : 0;
}

void PersistentObject::close() {
    mapper_.reset();
}

void PersistentObject::initialize_storage(uint64_t size_in_bytes,
                                   const std::string& backing_file,
                                   const std::string& fallback_prefix,
                                   uint64_t min_size_bytes,
                                   pycauset::MatrixType matrix_type,
                                   pycauset::DataType data_type,
                                   uint64_t rows,
                                   uint64_t cols) {
    uint64_t final_size = size_in_bytes;
    if (final_size < min_size_bytes) {
        final_size = min_size_bytes;
    }
    
    std::string path;
    if (backing_file.empty()) {
        // Check if we should use RAM-backed storage
        if (final_size <= pycauset::get_memory_threshold()) {
            path = ":memory:";
        } else {
            path = resolve_backing_file(backing_file, fallback_prefix);
        }
    } else {
        path = resolve_backing_file(backing_file, fallback_prefix);
    }
    
    // Create new file with header space
    mapper_ = std::make_unique<MemoryMapper>(path, final_size, true);

    // Initialize Header
    auto* header = mapper_->get_header();
    std::memset(header, 0, sizeof(pycauset::FileHeader));
    std::memcpy(header->magic, "PYCAUSET", 8);
    header->version = 2;
    header->matrix_type = matrix_type;
    header->data_type = data_type;
    header->rows = rows;
    header->cols = cols;
    header->scalar = 1.0;
    // If no backing file was requested, this is an auto-generated temporary file.
    // RAM-backed files are always temporary.
    header->is_temporary = (backing_file.empty() || path == ":memory:") ? 1 : 0;
    scalar_ = 1.0;
}

std::string PersistentObject::copy_storage(const std::string& result_file_hint) const {
    std::string source_path = get_backing_file();
    
    if (mapper_) {
        mapper_->flush();
    }

    std::string dest_path = resolve_backing_file(result_file_hint, "copy");
    
    if (source_path == ":memory:") {
        // For RAM-backed objects, we must write the memory buffer to disk manually
        std::ofstream outfile(dest_path, std::ios::binary);
        if (!outfile) {
            throw std::runtime_error("Failed to open destination file for writing: " + dest_path);
        }
        
        const char* data = static_cast<const char*>(static_cast<const void*>(mapper_->get_header()));
        size_t size = sizeof(pycauset::FileHeader) + mapper_->get_data_size();
        
        if (!outfile.write(data, size)) {
            throw std::runtime_error("Failed to write RAM object to disk: " + dest_path);
        }
        outfile.close();
    } else {
        if (source_path.empty()) {
            throw std::runtime_error("Cannot copy object without backing file");
        }
        // Use filesystem copy
        try {
            std::filesystem::copy_file(source_path, dest_path, std::filesystem::copy_options::overwrite_existing);
        } catch (const std::filesystem::filesystem_error& e) {
            throw std::runtime_error("Failed to copy backing file: " + std::string(e.what()));
        }
    }
    
    return dest_path;
}

void PersistentObject::set_scalar(double s) {
    scalar_ = s;
    if (mapper_) {
        mapper_->get_header()->scalar = s;
    }
}

uint64_t PersistentObject::get_seed() const {
    if (mapper_) {
        return mapper_->get_header()->seed;
    }
    return 0;
}

void PersistentObject::set_seed(uint64_t seed) {
    if (mapper_) {
        mapper_->get_header()->seed = seed;
    }
}

bool PersistentObject::is_temporary() const {
    if (mapper_) {
        return mapper_->get_header()->is_temporary != 0;
    }
    return false;
}

void PersistentObject::set_temporary(bool temp) {
    if (mapper_) {
        mapper_->get_header()->is_temporary = temp ? 1 : 0;
    }
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
