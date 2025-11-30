#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "MemoryMapper.hpp"
#include "FileFormat.hpp"

class PersistentObject {
public:
    PersistentObject();
    virtual ~PersistentObject();

    std::string get_backing_file() const;
    void close();

    double get_scalar() const { return scalar_; }
    void set_scalar(double s); // Updates header and member
    
    uint64_t get_seed() const;
    void set_seed(uint64_t seed);

    bool is_temporary() const;
    void set_temporary(bool temp);

    // Creates a copy of the backing file and returns the new path
    std::string copy_storage(const std::string& result_file_hint) const;

protected:
    // Constructor for loading from existing mapper
    explicit PersistentObject(std::unique_ptr<MemoryMapper> mapper);

    void initialize_storage(uint64_t size_in_bytes,
                           const std::string& backing_file,
                           const std::string& fallback_prefix,
                           uint64_t min_size_bytes,
                           pycauset::MatrixType matrix_type,
                           pycauset::DataType data_type,
                           uint64_t rows,
                           uint64_t cols);

    MemoryMapper* require_mapper();
    const MemoryMapper* require_mapper() const;
    
    static std::string resolve_backing_file(const std::string& requested,
                                          const std::string& fallback_prefix);

    double scalar_ = 1.0;
    std::unique_ptr<MemoryMapper> mapper_;
};
