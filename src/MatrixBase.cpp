#include "MatrixBase.hpp"

#include <stdexcept>
#include <cstring>
#include <filesystem>

#include "StoragePaths.hpp"

MatrixBase::MatrixBase(uint64_t n) : n_(n), scalar_(1.0) {}

MatrixBase::~MatrixBase() {
    if (mapper_) {
        bool temp = false;
        try {
            temp = is_temporary();
        } catch (...) {}

        std::string path = mapper_->get_filename();
        mapper_.reset(); // Close file mapping

        if (temp && !path.empty()) {
            try {
                std::filesystem::remove(path);
            } catch (...) {
                // Best effort deletion
            }
        }
    }
}

MatrixBase::MatrixBase(uint64_t n, std::unique_ptr<MemoryMapper> mapper) 
    : n_(n), mapper_(std::move(mapper)) {
    if (mapper_) {
        auto* header = mapper_->get_header();
        if (header->version >= 2) {
            scalar_ = header->scalar;
        } else {
            scalar_ = 1.0;
        }
    }
}

std::string MatrixBase::get_backing_file() const {
    if (mapper_) {
        return mapper_->get_filename();
    }
    return "";
}

void MatrixBase::close() {
    mapper_.reset();
}

void MatrixBase::initialize_storage(uint64_t size_in_bytes,
                                   const std::string& backing_file,
                                   const std::string& fallback_prefix,
                                   uint64_t min_size_bytes,
                                   pycauset::MatrixType matrix_type,
                                   pycauset::DataType data_type) {
    uint64_t final_size = size_in_bytes;
    if (final_size < min_size_bytes) {
        final_size = min_size_bytes;
    }
    std::string path = resolve_backing_file(backing_file, fallback_prefix);
    
    // Create new file with header space
    mapper_ = std::make_unique<MemoryMapper>(path, final_size, true);

    // Initialize Header
    auto* header = mapper_->get_header();
    std::memset(header, 0, sizeof(pycauset::FileHeader));
    std::memcpy(header->magic, "PYCAUSET", 8);
    header->version = 2;
    header->matrix_type = matrix_type;
    header->data_type = data_type;
    header->rows = n_;
    header->cols = n_;
    header->scalar = 1.0;
    // If no backing file was requested, this is an auto-generated temporary file.
    header->is_temporary = backing_file.empty() ? 1 : 0;
    scalar_ = 1.0;
}

std::string MatrixBase::copy_storage(const std::string& result_file_hint) const {
    std::string source_path = get_backing_file();
    if (source_path.empty()) {
        throw std::runtime_error("Cannot copy matrix without backing file");
    }
    
    if (mapper_) {
        mapper_->flush();
    }

    std::string dest_path = resolve_backing_file(result_file_hint, "copy");
    
    // Use filesystem copy
    try {
        std::filesystem::copy_file(source_path, dest_path, std::filesystem::copy_options::overwrite_existing);
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to copy backing file: " + std::string(e.what()));
    }
    
    return dest_path;
}

void MatrixBase::set_scalar(double s) {
    scalar_ = s;
    if (mapper_) {
        mapper_->get_header()->scalar = s;
    }
}

uint64_t MatrixBase::get_seed() const {
    if (mapper_) {
        return mapper_->get_header()->seed;
    }
    return 0;
}

void MatrixBase::set_seed(uint64_t seed) {
    if (mapper_) {
        mapper_->get_header()->seed = seed;
    }
}

bool MatrixBase::is_temporary() const {
    if (mapper_) {
        return mapper_->get_header()->is_temporary != 0;
    }
    return false;
}

void MatrixBase::set_temporary(bool temp) {
    if (mapper_) {
        mapper_->get_header()->is_temporary = temp ? 1 : 0;
    }
}

MemoryMapper* MatrixBase::require_mapper() {
    if (!mapper_) {
        throw std::runtime_error("Matrix backing file has been closed.");
    }
    return mapper_.get();
}

const MemoryMapper* MatrixBase::require_mapper() const {
    if (!mapper_) {
        throw std::runtime_error("Matrix backing file has been closed.");
    }
    return mapper_.get();
}

std::string MatrixBase::resolve_backing_file(const std::string& requested,
                                           const std::string& fallback_prefix) {
    if (!requested.empty()) {
        return requested;
    }
    return make_unique_storage_file(fallback_prefix);
}
