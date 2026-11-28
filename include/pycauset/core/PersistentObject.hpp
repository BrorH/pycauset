#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <complex>

#include "pycauset/core/Types.hpp"
#include "pycauset/core/MemoryHints.hpp"
#include "pycauset/core/IOAccelerator.hpp"

class PersistentObject {
public:
    // The storage; throws std::invalid_argument (ValueError) after close(). Every reader goes
    // through this: the NumPy export dereferenced the mapper directly and segfaulted on a
    // closed matrix (review 3.7 C-1).
    MemoryMapper* require_mapper();
    const MemoryMapper* require_mapper() const;

    // Constructor for loading/creating with explicit metadata
    PersistentObject(std::shared_ptr<MemoryMapper> mapper,
                     pycauset::Structure structure,
                     pycauset::DataType data_type,
                     uint64_t rows,
                     uint64_t cols,
                     uint64_t seed = 0,
                     std::complex<double> scalar = 1.0,
                     bool is_transposed = false);

    virtual ~PersistentObject();

    virtual std::unique_ptr<PersistentObject> clone() const = 0;

    std::string get_backing_file() const;
    void close();

    // --- Tiered Storage Management ---
    
    /**
     * @brief Ensures that this object has exclusive access to its storage.
     * If the storage is shared (CoW), this triggers a deep copy.
     */
    void ensure_unique();

    // The storage itself (shared, copy-on-write). Lets another object view the same bytes
    // with a different shape, e.g. a dense vector as a 1 x n matrix (math/VectorOps.hpp).
    const std::shared_ptr<MemoryMapper>& shared_mapper() const { return mapper_; }

    // --- Metadata Accessors ---

    std::complex<double> get_scalar() const { return scalar_; }
    void set_scalar(std::complex<double> s); 
    
    uint64_t get_seed() const { return seed_; }
    void set_seed(uint64_t seed);

    // Whether the storage is deleted with its last owner (auto-named files, copies, RAM). It
    // belongs to the storage, so every view of it agrees (MemoryMapper::is_temporary).
    bool is_temporary() const;
    void set_temporary(bool temp);

    bool is_transposed() const { return is_transposed_; }
    void set_transposed(bool transposed);

    bool is_conjugated() const { return is_conjugated_; }
    void set_conjugated(bool conjugated);

    pycauset::DataType get_data_type() const { return data_type_; }
    pycauset::Structure get_structure() const { return structure_; }

    
    uint64_t get_rows() const { return rows_; }
    uint64_t get_cols() const { return cols_; }

    // Creates a copy of the backing file and returns the new path
    std::string copy_storage(const std::string& result_file_hint) const;

    // Access to IO Accelerator
    pycauset::core::IOAccelerator* get_accelerator() const;

    /**
     * @brief Sends a memory access hint to the IO Accelerator.
     * This is part of the "Lookahead Protocol".
     */
    void hint(const pycauset::core::MemoryHint& hint) const;

protected:
    // Default constructor for subclasses that initialize later
    PersistentObject();

    void initialize_storage(uint64_t size_in_bytes,
                           const std::string& backing_file,
                           const std::string& fallback_prefix,
                           uint64_t min_size_bytes,
                           pycauset::Structure structure,
                           pycauset::DataType data_type,
                           uint64_t rows,
                           uint64_t cols,
                           size_t offset = 0,
                           bool create_new = true);

    void copy_storage_now();  // replace the storage by a private copy (RAM or a temp file)

    
    static std::string resolve_backing_file(const std::string& requested,
                                          const std::string& fallback_prefix);

    std::shared_ptr<MemoryMapper> mapper_;
    
    // Metadata members
    pycauset::Structure structure_ = pycauset::Structure::UNKNOWN;
    pycauset::DataType data_type_ = pycauset::DataType::UNKNOWN;
    uint64_t rows_ = 0;
    uint64_t cols_ = 0;
    uint64_t seed_ = 0;
    std::complex<double> scalar_ = 1.0;
    bool is_transposed_ = false;
    bool is_conjugated_ = false;

    // IO Accelerator
    mutable std::unique_ptr<pycauset::core::IOAccelerator> accelerator_;
};

