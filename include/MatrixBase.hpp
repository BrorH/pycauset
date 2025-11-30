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
    std::string get_backing_file() const;
    void close();

    double get_scalar() const { return scalar_; }
    void set_scalar(double s); // Updates header and member
    uint64_t get_seed() const;

    bool is_temporary() const;
    void set_temporary(bool temp);

    virtual double get_element_as_double(uint64_t i, uint64_t j) const = 0;

    // Creates a copy of the backing file and returns the new path
    std::string copy_storage(const std::string& result_file_hint) const;

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
    double scalar_ = 1.0;
    std::unique_ptr<MemoryMapper> mapper_;

private:
    static std::string resolve_backing_file(const std::string& requested,
                                          const std::string& fallback_prefix);
};
