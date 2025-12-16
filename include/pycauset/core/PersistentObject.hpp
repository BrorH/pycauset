#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <complex>

#include "pycauset/core/Types.hpp"
#include "pycauset/core/MemoryHints.hpp"
#include "pycauset/core/IOAccelerator.hpp"

// class MemoryMapper; // Already included in IOAccelerator.hpp? No, IOAccelerator.hpp includes MemoryMapper.hpp.
// But PersistentObject.hpp uses MemoryMapper* in constructor.

namespace pycauset {
namespace core {

// class IOAccelerator; // Forward declaration removed

enum class StorageState {
    RAM_ONLY,       // Data is in anonymous memory (RAM)
    DISK_BACKED,    // Data is memory-mapped from a file
    TRANSITIONING   // Currently moving between states
};

} // namespace core
} // namespace pycauset

class PersistentObject {
public:
    // Constructor for loading/creating with explicit metadata
    PersistentObject(std::shared_ptr<MemoryMapper> mapper,
                     pycauset::MatrixType matrix_type,
                     pycauset::DataType data_type,
                     uint64_t rows,
                     uint64_t cols,
                     uint64_t seed = 0,
                     std::complex<double> scalar = 1.0,
                     bool is_transposed = false,
                     bool is_temporary = false);

    virtual ~PersistentObject() {
    }

    virtual std::unique_ptr<PersistentObject> clone() const = 0;

    std::string get_backing_file() const;
    void close();

    // --- Tiered Storage Management ---
    
    /**
     * @brief Ensures that this object has exclusive access to its storage.
     * If the storage is shared (CoW), this triggers a deep copy.
     */
    void ensure_unique();

    /**
     * @brief Notifies the MemoryGovernor that this object is being accessed.
     * Should be called by Solvers before heavy operations.
     */
    void touch();

    /**
     * @brief Moves the object from RAM to Disk to free up memory.
     * @return true if successful, false if already on disk or failed.
     */
    bool spill_to_disk();

    /**
     * @brief Moves the object from Disk to RAM for performance.
     * @return true if successful, false if RAM is full or already in RAM.
     */
    bool promote_to_ram();

    /**
     * @brief Returns the current storage state.
     */
    pycauset::core::StorageState get_storage_state() const { return storage_state_; }

    // --- Metadata Accessors ---

    std::complex<double> get_scalar() const { return scalar_; }
    void set_scalar(std::complex<double> s); 
    
    uint64_t get_seed() const { return seed_; }
    void set_seed(uint64_t seed);

    bool is_temporary() const { return is_temporary_; }
    void set_temporary(bool temp);

    bool is_transposed() const { return is_transposed_; }
    void set_transposed(bool transposed);

    bool is_conjugated() const { return is_conjugated_; }
    void set_conjugated(bool conjugated);

    pycauset::DataType get_data_type() const { return data_type_; }
    pycauset::MatrixType get_matrix_type() const { return matrix_type_; }
    
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

    std::shared_ptr<MemoryMapper> mapper_;
    
    // Metadata members
    pycauset::MatrixType matrix_type_ = pycauset::MatrixType::UNKNOWN;
    pycauset::DataType data_type_ = pycauset::DataType::UNKNOWN;
    uint64_t rows_ = 0;
    uint64_t cols_ = 0;
    uint64_t seed_ = 0;
    std::complex<double> scalar_ = 1.0;
    bool is_transposed_ = false;
    bool is_conjugated_ = false;
    bool is_temporary_ = false;

    // Tiered Storage State
    pycauset::core::StorageState storage_state_ = pycauset::core::StorageState::DISK_BACKED; // Default for legacy compatibility

    // IO Accelerator
    mutable std::unique_ptr<pycauset::core::IOAccelerator> accelerator_;
};

