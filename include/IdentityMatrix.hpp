#pragma once

#include "MatrixBase.hpp"
#include "MatrixTraits.hpp"
#include "StoragePaths.hpp"
#include <stdexcept>
#include <string>

class IdentityMatrix : public MatrixBase {
public:
    IdentityMatrix(uint64_t n, const std::string& backing_file = "")
        : MatrixBase(n) {
        // Identity matrix only needs the header, no data segment.
        initialize_storage(0, backing_file, 
                         "identity", 
                         0,
                         pycauset::MatrixType::IDENTITY, 
                         pycauset::DataType::FLOAT64,
                         n, n);
    }

    IdentityMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
        : MatrixBase(n, std::move(mapper)) {}

    double get_element_as_double(uint64_t i, uint64_t j) const override {
        if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");
        if (i == j) {
            return scalar_;
        }
        return 0.0;
    }

    // Specialized operations returning IdentityMatrix
    std::unique_ptr<IdentityMatrix> add(const IdentityMatrix& other, const std::string& result_file = "") const {
        if (n_ != other.size()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix>(n_, result_file);
        result->set_scalar(scalar_ + other.get_scalar());
        return result;
    }

    std::unique_ptr<IdentityMatrix> subtract(const IdentityMatrix& other, const std::string& result_file = "") const {
        if (n_ != other.size()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix>(n_, result_file);
        result->set_scalar(scalar_ - other.get_scalar());
        return result;
    }

    // Matrix multiplication of two scaled identity matrices is a scaled identity matrix
    // (s1*I) * (s2*I) = (s1*s2)*I
    std::unique_ptr<IdentityMatrix> multiply(const IdentityMatrix& other, const std::string& result_file = "") const {
        if (n_ != other.size()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix>(n_, result_file);
        result->set_scalar(scalar_ * other.get_scalar());
        return result;
    }
    
    // Elementwise multiplication of two scaled identity matrices is a scaled identity matrix
    // (s1*I) .* (s2*I) has diagonal s1*s2 and off-diagonal 0*0=0.
    std::unique_ptr<IdentityMatrix> elementwise_multiply(const IdentityMatrix& other, const std::string& result_file = "") const {
        if (n_ != other.size()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix>(n_, result_file);
        result->set_scalar(scalar_ * other.get_scalar());
        return result;
    }

    std::unique_ptr<IdentityMatrix> multiply_scalar(double factor, const std::string& result_file = "") const {
        auto result = std::make_unique<IdentityMatrix>(n_, result_file);
        result->set_scalar(scalar_ * factor);
        return result;
    }
};
