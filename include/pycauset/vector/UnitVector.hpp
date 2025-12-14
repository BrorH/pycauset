#pragma once

#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>

namespace pycauset {

class UnitVector : public VectorBase {
public:
    // Constructor
    UnitVector(uint64_t n, uint64_t active_index, const std::string& backing_file = "")
        : VectorBase(n, 
                     0, // No storage
                     backing_file, 
                     "unit_vector", 
                     pycauset::MatrixType::UNIT_VECTOR, 
                     pycauset::DataType::FLOAT64), // Unit vectors are conceptually float/double
          active_index_(active_index) {
        
        if (active_index >= n) {
            throw std::out_of_range("Active index out of bounds");
        }
        // We store the active index in the seed field for persistence
        set_seed(active_index);
    }

    // Constructor for loading
    UnitVector(uint64_t n, 
               uint64_t active_index,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed)
        : VectorBase(n, 
                     0, // No storage
                     backing_file, 
                     offset,
                     seed,
                     scalar,
                     is_transposed,
                     pycauset::MatrixType::UNIT_VECTOR, 
                     pycauset::DataType::FLOAT64),
          active_index_(active_index) {
    }

    UnitVector(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : VectorBase(n, std::move(mapper), pycauset::MatrixType::UNIT_VECTOR, pycauset::DataType::FLOAT64) {
        // Recover active index from seed
        active_index_ = get_seed();
    }

    double get_element_as_double(uint64_t i) const override {
        if (i >= n_) throw std::out_of_range("Index out of bounds");
        if (i == active_index_) {
            return scalar_.real();
        }
        return 0.0;
    }

    std::unique_ptr<VectorBase> transpose(const std::string& saveas = "") const override {
        auto result = std::make_unique<UnitVector>(n_, active_index_, saveas);
        result->set_scalar(scalar_);
        result->set_transposed(!is_transposed());
        return result;
    }

    std::unique_ptr<VectorBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        auto result = std::make_unique<UnitVector>(n_, active_index_, result_file);
        result->set_scalar(scalar_ * factor);
        result->set_transposed(is_transposed());
        return result;
    }

    std::unique_ptr<VectorBase> add_scalar(double scalar, const std::string& result_file = "") const override {
        auto result = std::make_unique<DenseVector<double>>(n_, result_file);
        double* ptr = result->data();
        double background = scalar;
        double active_val = scalar_.real() + scalar;
        
        std::fill_n(ptr, n_, background);
        if (active_index_ < n_) {
            ptr[active_index_] = active_val;
        }
        
        if (is_transposed()) {
            result->set_transposed(true);
        }
        return result;
    }
    
    uint64_t get_active_index() const { return active_index_; }

    // Specialized operations
    std::unique_ptr<UnitVector> add(const UnitVector& other, const std::string& result_file = "") const {
        if (n_ != other.size()) throw std::invalid_argument("Dimension mismatch");
        if (active_index_ != other.get_active_index()) {
            throw std::invalid_argument("Cannot add UnitVectors with different indices (result is not a UnitVector)");
        }
        
        auto result = std::make_unique<UnitVector>(n_, active_index_, result_file);
        result->set_scalar(scalar_ + other.get_scalar());
        return result;
    }

private:
    uint64_t active_index_;
};

} // namespace pycauset
