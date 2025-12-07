#pragma once

#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include <stdexcept>
#include <string>

namespace pycauset {

template <typename T>
class IdentityMatrix : public DiagonalMatrix<T> {
public:
    IdentityMatrix(uint64_t n, const std::string& backing_file = "")
        : DiagonalMatrix<T>(n, 
                            0, // No storage
                            backing_file, 
                            "identity", 
                            pycauset::MatrixType::IDENTITY, 
                            MatrixTraits<T>::data_type) {
    }

    // Constructor for loading
    IdentityMatrix(uint64_t n, 
                   const std::string& backing_file,
                   size_t offset,
                   uint64_t seed,
                   double scalar,
                   bool is_transposed)
        : DiagonalMatrix<T>(n, 
                            0, // No storage
                            backing_file, 
                            offset,
                            seed,
                            scalar,
                            is_transposed,
                            pycauset::MatrixType::IDENTITY, 
                            MatrixTraits<T>::data_type) {
    }

    IdentityMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
        : DiagonalMatrix<T>(n, std::move(mapper), pycauset::MatrixType::IDENTITY, MatrixTraits<T>::data_type) {}

    // Override set/get to avoid accessing data()
    void set(uint64_t i, uint64_t j, T value) override {
        throw std::runtime_error("Cannot modify IdentityMatrix elements");
    }

    T get(uint64_t i, uint64_t j) const override {
        if (i >= this->n_ || j >= this->n_) throw std::out_of_range("Index out of bounds");
        if (i == j) return static_cast<T>(1);
        return static_cast<T>(0);
    }

    void set_diagonal(uint64_t i, T value) override {
        throw std::runtime_error("Cannot modify IdentityMatrix elements");
    }

    T get_diagonal(uint64_t i) const override {
        if (i >= this->n_) throw std::out_of_range("Index out of bounds");
        return static_cast<T>(1);
    }

    double get_element_as_double(uint64_t i, uint64_t j) const override {
        if (i >= this->n_ || j >= this->n_) throw std::out_of_range("Index out of bounds");
        if (i == j) {
            return this->scalar_;
        }
        return 0.0;
    }

    // Specialized operations returning IdentityMatrix
    std::unique_ptr<IdentityMatrix<T>> add(const IdentityMatrix<T>& other, const std::string& result_file = "") const {
        if (this->n_ != other.size()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix<T>>(this->n_, result_file);
        result->set_scalar(this->scalar_ + other.get_scalar());
        return result;
    }

    std::unique_ptr<IdentityMatrix<T>> subtract(const IdentityMatrix<T>& other, const std::string& result_file = "") const {
        if (this->n_ != other.size()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix<T>>(this->n_, result_file);
        result->set_scalar(this->scalar_ - other.get_scalar());
        return result;
    }

    std::unique_ptr<IdentityMatrix<T>> multiply(const IdentityMatrix<T>& other, const std::string& result_file = "") const {
        if (this->n_ != other.size()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix<T>>(this->n_, result_file);
        result->set_scalar(this->scalar_ * other.get_scalar());
        return result;
    }
    
    std::unique_ptr<IdentityMatrix<T>> elementwise_multiply(const IdentityMatrix<T>& other, const std::string& result_file = "") const {
        if (this->n_ != other.size()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix<T>>(this->n_, result_file);
        result->set_scalar(this->scalar_ * other.get_scalar());
        return result;
    }

    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override {
        auto result = std::make_unique<IdentityMatrix<T>>(this->n_, result_file);
        result->set_scalar(this->scalar_);
        result->set_transposed(!this->is_transposed());
        return result;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        auto result = std::make_unique<IdentityMatrix<T>>(this->n_, result_file);
        result->set_scalar(this->scalar_ * factor);
        if (result_file.empty()) {
            result->set_temporary(true);
        }
        return result;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    std::unique_ptr<MatrixBase> add_scalar(double scalar, const std::string& result_file = "") const override {
        auto result = std::make_unique<DenseMatrix<double>>(this->n_, result_file);
        double* dst_data = result->data();
        
        for (uint64_t i = 0; i < this->n_; ++i) {
            for (uint64_t j = 0; j < this->n_; ++j) {
                if (i == j) {
                    dst_data[i * this->n_ + j] = this->scalar_ + scalar;
                } else {
                    dst_data[i * this->n_ + j] = scalar;
                }
            }
        }
        
        if (result_file.empty()) {
            result->set_temporary(true);
        }
        return result;
    }
    
    std::unique_ptr<MatrixBase> add_scalar(int64_t scalar, const std::string& result_file = "") const override {
        return add_scalar(static_cast<double>(scalar), result_file);
    }
};

} // namespace pycauset
