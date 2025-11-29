#include "MatrixBase.hpp"

#include <stdexcept>
#include <cstring>

#include "StoragePaths.hpp"

MatrixBase::MatrixBase(uint64_t n) : n_(n) {}

MatrixBase::MatrixBase(uint64_t n, std::unique_ptr<MemoryMapper> mapper) 
    : n_(n), mapper_(std::move(mapper)) {}

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
    header->version = 1;
    header->matrix_type = matrix_type;
    header->data_type = data_type;
    header->rows = n_;
    header->cols = n_;
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
