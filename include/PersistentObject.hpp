#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "MemoryMapper.hpp"
#include "FileFormat.hpp"

class PersistentObject {
public:
    // Constructor for loading/creating with explicit metadata
    PersistentObject(std::unique_ptr<MemoryMapper> mapper,
                     pycauset::MatrixType matrix_type,
                     pycauset::DataType data_type,
                     uint64_t rows,
                     uint64_t cols,
                     uint64_t seed = 0,
                     double scalar = 1.0,
                     bool is_transposed = false,
                     bool is_temporary = false);

    virtual ~PersistentObject();

    std::string get_backing_file() const;
    void close();

    double get_scalar() const { return scalar_; }
    void set_scalar(double s); 
    
    uint64_t get_seed() const { return seed_; }
    void set_seed(uint64_t seed);

    bool is_temporary() const { return is_temporary_; }
    void set_temporary(bool temp);

    bool is_transposed() const { return is_transposed_; }
    void set_transposed(bool transposed);

    pycauset::DataType get_data_type() const { return data_type_; }
    pycauset::MatrixType get_matrix_type() const { return matrix_type_; }
    
    uint64_t get_rows() const { return rows_; }
    uint64_t get_cols() const { return cols_; }

    // Creates a copy of the backing file and returns the new path
    std::string copy_storage(const std::string& result_file_hint) const;

protected:
    // Default constructor for subclasses that initialize later
    PersistentObject();

    void initialize_storage(uint64_t size_in_bytes,
                           const std::string& backing_file,
                           const std::string& fallback_prefix,
                           uint64_t min_size_bytes,
                           pycauset::MatrixType matrix_type,
                           pycauset::DataType data_type,
                           uint64_t rows,
                           uint64_t cols,
                           size_t offset = 0,
                           bool create_new = true);

    MemoryMapper* require_mapper();
    const MemoryMapper* require_mapper() const;
    
    static std::string resolve_backing_file(const std::string& requested,
                                          const std::string& fallback_prefix);

    std::unique_ptr<MemoryMapper> mapper_;
    
    // Metadata members
    pycauset::MatrixType matrix_type_ = pycauset::MatrixType::UNKNOWN;
    pycauset::DataType data_type_ = pycauset::DataType::UNKNOWN;
    uint64_t rows_ = 0;
    uint64_t cols_ = 0;
    uint64_t seed_ = 0;
    double scalar_ = 1.0;
    bool is_transposed_ = false;
    bool is_temporary_ = false;
};
