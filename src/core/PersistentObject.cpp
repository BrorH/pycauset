#include "pycauset/core/PersistentObject.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/IOAccelerator.hpp"

#include <stdexcept>
#include <cstring>

#include "pycauset/core/StorageUtils.hpp"

PersistentObject::PersistentObject() = default;

PersistentObject::PersistentObject(std::shared_ptr<MemoryMapper> mapper,
                                   pycauset::Structure structure,
                                   pycauset::DataType data_type,
                                   uint64_t rows,
                                   uint64_t cols,
                                   uint64_t seed,
                                   std::complex<double> scalar,
                                   bool is_transposed)
    : mapper_(std::move(mapper)),
      structure_(structure),
      data_type_(data_type),
      rows_(rows),
      cols_(cols),
      seed_(seed),
      scalar_(scalar),
      is_transposed_(is_transposed) {}

PersistentObject::~PersistentObject() = default;

std::string PersistentObject::get_backing_file() const {
    if (mapper_) {
        return mapper_->get_filename();
    }
    return "";
}

void PersistentObject::close() {
    // The storage (and its temporary file, if any) goes with its last owner: other views of
    // it keep it alive.
    mapper_.reset();
    accelerator_.reset();
}

void PersistentObject::ensure_unique() {
    if (!mapper_) return;
    if (mapper_.use_count() > 1) copy_storage_now();
    // The write gate: every writer calls this before changing bytes, so the derived values
    // of this storage are dropped here (DERIVED_CACHE.md invariant 2). After a copy the new
    // storage has none; the old one keeps its own (its bytes do not change).
    mapper_->bump_version();
}

// Placement of a new storage: RAM when under the memory threshold and it fits, otherwise
// an auto-named temporary file. One rule for new objects and copies (review 3.1 C-13: a
// copy of a loaded snapshot went to a file that was never deleted, however small).
static bool place_in_ram(uint64_t bytes) {
    return bytes <= pycauset::get_memory_threshold() && pycauset::core::MemoryGovernor::instance().can_fit_in_ram(bytes);
}

void PersistentObject::copy_storage_now() {
    const size_t size = mapper_->get_data_size();
    const bool use_ram = place_in_ram(size);
    auto copy = std::make_shared<MemoryMapper>(use_ram ? std::string(":memory:") : pycauset::make_unique_storage_file("cow_copy"),
                                               size, 0, true);
    if (!use_ram) copy->set_temporary(true);
    if (size > 0) std::memcpy(copy->get_data(), mapper_->get_data(), size);  // empty: null pointers
    mapper_ = std::move(copy);
    accelerator_.reset();
}

void PersistentObject::initialize_storage(uint64_t size_in_bytes,
                                   const std::string& backing_file,
                                   const std::string& fallback_prefix,
                                   uint64_t min_size_bytes,
                                   pycauset::Structure structure,
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
    bool use_ram = false;
    if (backing_file.empty() || backing_file == ":memory:") {
        use_ram = place_in_ram(final_size);
        path = use_ram ? std::string(":memory:") : resolve_backing_file("", fallback_prefix);
    } else {
        path = resolve_backing_file(backing_file, fallback_prefix);
    }

    // Create new file, no header space needed
    // An existing file is a saved snapshot: map it copy-on-write, so writes never reach it (F-093).
    mapper_ = std::make_unique<MemoryMapper>(path, final_size, offset, create_new, /*private_writes=*/!create_new);
    accelerator_.reset();
    // Auto-named files and explicit `.tmp` names are deleted with the storage.
    if (create_new && !use_ram && (backing_file.empty() || backing_file.ends_with(".pycauset-tmp"))) mapper_->set_temporary(true);

#ifdef _WIN32
    // Prefetch a new file that fits in RAM to populate page tables before it is written
    // (Linux: MAP_POPULATE in MemoryMapper, same condition).
    if (create_new && !use_ram && pycauset::core::MemoryGovernor::instance().can_fit_in_ram(final_size)) {
        get_accelerator()->prefetch(0, final_size);
    }
#endif

    // Initialize members
    structure_ = structure;
    data_type_ = data_type;
    rows_ = rows;
    cols_ = cols;
    scalar_ = 1.0;
    is_transposed_ = false;
    seed_ = 0;
}


std::string PersistentObject::copy_storage(const std::string& result_file_hint) const {
    const MemoryMapper* src = require_mapper();
    std::string dest_path = resolve_backing_file(result_file_hint, "copy");
    // Copies the bytes in memory into a new file (with its header): RAM objects and payloads
    // inside a container (offset > 0) have no file of their own to copy.
    try {
        const size_t size = src->get_data_size();
        MemoryMapper dest(dest_path, size, 0, true);
        if (size > 0) {  // empty objects have null data pointers (F-025)
            std::memcpy(dest.get_data(), src->get_data(), size);
        }
        dest.flush();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to copy storage to " + dest_path + ": " + e.what());
    }
    return dest_path;
}

void PersistentObject::set_scalar(std::complex<double> s) {
    scalar_ = s;
}

void PersistentObject::set_seed(uint64_t seed) {
    seed_ = seed;
}

bool PersistentObject::is_temporary() const {
    return mapper_ && mapper_->is_temporary();
}

void PersistentObject::set_temporary(bool temp) {
    if (mapper_) mapper_->set_temporary(temp);
}

void PersistentObject::set_transposed(bool transposed) {
    is_transposed_ = transposed;
}

void PersistentObject::set_conjugated(bool conjugated) {
    is_conjugated_ = conjugated;
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

pycauset::core::IOAccelerator* PersistentObject::get_accelerator() const {
    if (!accelerator_) {
        if (mapper_) {
            accelerator_ = std::make_unique<pycauset::core::IOAccelerator>(mapper_.get());
        }
    }
    return accelerator_.get();
}

void PersistentObject::hint(const pycauset::core::MemoryHint& hint) const {
    if (auto* acc = get_accelerator()) {
        acc->process_hint(hint);
    }
}

