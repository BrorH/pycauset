#pragma once

#include "MatrixBase.hpp"
#include "MatrixTraits.hpp"
#include "StoragePaths.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <type_traits>

template <typename T>
class DenseMatrix : public MatrixBase {
public:
    DenseMatrix(uint64_t n, const std::string& backing_file = "")
        : MatrixBase(n) {
        uint64_t size_in_bytes = n * n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         std::string("dense_") + MatrixTraits<T>::name, 
                         sizeof(T),
                         pycauset::MatrixType::DENSE_FLOAT, // We might need to adjust this enum if we want DENSE_INT
                         MatrixTraits<T>::data_type,
                         n, n);
        
        // Note: MatrixType::DENSE_FLOAT is currently the only dense type in the enum.
        // If we want to support Dense Integer matrices properly in the file format, 
        // we might need to add DENSE_INTEGER to the enum or reuse DENSE_FLOAT but check DataType.
        // For now, we'll use DENSE_FLOAT as a placeholder for "Dense" structure.
    }

    DenseMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
        : MatrixBase(n, std::move(mapper)) {}

    void set(uint64_t i, uint64_t j, T value) {
        if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");
        data()[i * n_ + j] = value;
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");
        return data()[i * n_ + j];
    }

    double get_element_as_double(uint64_t i, uint64_t j) const override {
        if (scalar_ == 1.0) {
            return static_cast<double>(get(i, j));
        }
        return static_cast<double>(get(i, j)) * scalar_;
    }

    T* data() { return static_cast<T*>(require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(require_mapper()->get_data()); }

    std::unique_ptr<DenseMatrix<T>> multiply_scalar(double factor, const std::string& result_file = "") const {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto new_matrix = std::make_unique<DenseMatrix<T>>(n_, std::move(mapper));
        new_matrix->set_scalar(scalar_ * factor);
        // If result_file was not specified, it's a temporary intermediate result
        if (result_file.empty()) {
            new_matrix->set_temporary(true);
        }
        return new_matrix;
    }

    std::unique_ptr<DenseMatrix<T>> bitwise_not(const std::string& result_file = "") const {
        auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);
        
        const T* src_data = data();
        T* dst_data = result->data();
        
        uint64_t total_elements = n_ * n_;
        
        // Treat as raw bits for bitwise not
        if constexpr (std::is_floating_point_v<T>) {
            const uint64_t* src_bits = reinterpret_cast<const uint64_t*>(src_data);
            uint64_t* dst_bits = reinterpret_cast<uint64_t*>(dst_data);
            // Assuming 64-bit double. For float it would be 32-bit.
            // This implementation assumes T=double.
            for (uint64_t i = 0; i < total_elements; ++i) {
                dst_bits[i] = ~src_bits[i];
            }
        } else {
            for (uint64_t i = 0; i < total_elements; ++i) {
                dst_data[i] = ~src_data[i];
            }
        }
        
        result->set_scalar(scalar_);
        return result;
    }

    std::unique_ptr<DenseMatrix<T>> multiply(const DenseMatrix<T>& other, const std::string& result_file = "") const {
        if (n_ != other.size()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);
        
        const T* a_data = data();
        const T* b_data = other.data();
        T* c_data = result->data();
        
        std::fill(c_data, c_data + n_ * n_, static_cast<T>(0));

        // IKJ algorithm
        for (uint64_t i = 0; i < n_; ++i) {
            for (uint64_t k = 0; k < n_; ++k) {
                T val_a = a_data[i * n_ + k];
                if (val_a == static_cast<T>(0)) continue;
                
                const T* b_row = b_data + k * n_;
                T* c_row = c_data + i * n_;
                
                for (uint64_t j = 0; j < n_; ++j) {
                    c_row[j] += val_a * b_row[j];
                }
            }
        }
        
        result->set_scalar(scalar_ * other.get_scalar());
        return result;
    }

    // Inverse is only implemented for double
    std::unique_ptr<DenseMatrix<double>> inverse(const std::string& result_file = "") const {
        if constexpr (!std::is_same_v<T, double>) {
            throw std::runtime_error("Inverse only supported for FloatMatrix (DenseMatrix<double>)");
        } else {
            // Implementation copied from KMatrix.cpp
            if (n_ == 0) return std::make_unique<DenseMatrix<double>>(0, result_file);
            if (scalar_ == 0.0) throw std::runtime_error("Matrix scalar is 0, cannot invert");

            std::string work_path = copy_storage(make_unique_storage_file("inverse_work"));
            auto work_mapper = std::make_unique<MemoryMapper>(work_path, 0, false);
            DenseMatrix<double> work(n_, std::move(work_mapper));
            work.set_temporary(true); // Ensure work file is deleted
            
            auto result = std::make_unique<DenseMatrix<double>>(n_, result_file);
            
            double* w = work.data();
            double* r = result->data();
            
            std::fill(r, r + n_ * n_, 0.0);
            for (uint64_t i = 0; i < n_; ++i) r[i * n_ + i] = 1.0;

            for (uint64_t i = 0; i < n_; ++i) {
                uint64_t pivot = i;
                double max_val = std::abs(w[i * n_ + i]);
                for (uint64_t k = i + 1; k < n_; ++k) {
                    double val = std::abs(w[k * n_ + i]);
                    if (val > max_val) {
                        max_val = val;
                        pivot = k;
                    }
                }

                if (max_val < 1e-12) {
                    work.close();
                    std::filesystem::remove(work_path);
                    throw std::runtime_error("Matrix is singular or nearly singular");
                }

                if (pivot != i) {
                    for (uint64_t j = 0; j < n_; ++j) {
                        std::swap(w[i * n_ + j], w[pivot * n_ + j]);
                        std::swap(r[i * n_ + j], r[pivot * n_ + j]);
                    }
                }

                double div = w[i * n_ + i];
                double inv_div = 1.0 / div;
                
                for (uint64_t j = 0; j < n_; ++j) {
                    w[i * n_ + j] *= inv_div;
                    r[i * n_ + j] *= inv_div;
                }

                for (uint64_t k = 0; k < n_; ++k) {
                    if (k != i) {
                        double factor = w[k * n_ + i];
                        for (uint64_t j = 0; j < n_; ++j) {
                            w[k * n_ + j] -= factor * w[i * n_ + j];
                            r[k * n_ + j] -= factor * r[i * n_ + j];
                        }
                    }
                }
            }
            
            if (scalar_ != 1.0) result->set_scalar(1.0 / scalar_);
            work.close();
            std::filesystem::remove(work_path);
            return result;
        }
    }
};
