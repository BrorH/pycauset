#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "MemoryMapper.hpp"
#include "FileFormat.hpp"

class MatrixBase {
public:
    explicit MatrixBase(uint64_t n);
    virtual ~MatrixBase() = default;

    uint64_t size() const { return n_; }
    void close();

protected:
    // Constructor for loading from existing mapper
    MatrixBase(uint64_t n, std::unique_ptr<MemoryMapper> mapper);

    void initialize_storage(uint64_t size_in_bytes,
                           const std::string& backing_file,
                           const std::string& fallback_prefix,
                           uint64_t min_size_bytes,
                           pycauset::MatrixType matrix_type,
                           pycauset::DataType data_type);

    MemoryMapper* require_mapper();
    const MemoryMapper* require_mapper() const;

    uint64_t n_;
    std::unique_ptr<MemoryMapper> mapper_;

private:
    static std::string resolve_backing_file(const std::string& requested,
                                          const std::string& fallback_prefix);
};
