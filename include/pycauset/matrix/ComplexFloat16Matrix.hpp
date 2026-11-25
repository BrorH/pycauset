#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/core/StorageUtils.hpp"

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace pycauset {

class ComplexFloat16Matrix : public DenseMatrixBase {
public:
    ComplexFloat16Matrix(uint64_t n, const std::string& backing_file = "")
        : ComplexFloat16Matrix(n, n, backing_file) {}

    ComplexFloat16Matrix(uint64_t rows, uint64_t cols, const std::string& backing_file = "")
        : DenseMatrixBase(rows, cols, pycauset::Structure::DENSE, pycauset::DataType::COMPLEX_FLOAT16) {
        const uint64_t plane_elems = rows * cols;
        const uint64_t size_in_bytes = 2ULL * plane_elems * sizeof(pycauset::float16_t);
        initialize_storage(
            size_in_bytes,
            backing_file,
            std::string("dense_complex_float16"),
            sizeof(pycauset::float16_t),
            pycauset::Structure::DENSE,
            pycauset::DataType::COMPLEX_FLOAT16,
            rows,
            cols);
    }

    ComplexFloat16Matrix(uint64_t n,
                         const std::string& backing_file,
                         size_t offset,
                         uint64_t seed,
                         std::complex<double> scalar,
                         bool is_transposed)
        : ComplexFloat16Matrix(n, n, backing_file, offset, seed, scalar, is_transposed) {}

    ComplexFloat16Matrix(uint64_t rows,
                         uint64_t cols,
                         const std::string& backing_file,
                         size_t offset,
                         uint64_t seed,
                         std::complex<double> scalar,
                         bool is_transposed)
        : DenseMatrixBase(rows, cols, pycauset::Structure::DENSE, pycauset::DataType::COMPLEX_FLOAT16) {
        const uint64_t plane_elems = rows * cols;
        const uint64_t size_in_bytes = 2ULL * plane_elems * sizeof(pycauset::float16_t);
        initialize_storage(
            size_in_bytes,
            backing_file,
            "",
            sizeof(pycauset::float16_t),
            pycauset::Structure::DENSE,
            pycauset::DataType::COMPLEX_FLOAT16,
            rows,
            cols,
            offset,
            false);

        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    ComplexFloat16Matrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : ComplexFloat16Matrix(n, n, std::move(mapper)) {}

    ComplexFloat16Matrix(uint64_t rows, uint64_t cols, std::shared_ptr<MemoryMapper> mapper)
        : DenseMatrixBase(rows, cols, std::move(mapper), pycauset::Structure::DENSE, pycauset::DataType::COMPLEX_FLOAT16) {}

    ~ComplexFloat16Matrix() override = default;

    pycauset::float16_t* real_data() { return static_cast<pycauset::float16_t*>(require_mapper()->get_data()); }
    const pycauset::float16_t* real_data() const {
        return static_cast<const pycauset::float16_t*>(require_mapper()->get_data());
    }

    pycauset::float16_t* imag_data() { return real_data() + (base_rows() * base_cols()); }
    const pycauset::float16_t* imag_data() const { return real_data() + (base_rows() * base_cols()); }

    void fill(std::complex<double> value) {
        ensure_unique();
        const uint64_t total = base_rows() * base_cols();
        auto* rdst = real_data();
        auto* idst = imag_data();
        const pycauset::float16_t rv(value.real());
        const pycauset::float16_t iv(value.imag());
        for (uint64_t idx = 0; idx < total; ++idx) {
            rdst[idx] = rv;
            idst[idx] = iv;
        }
    }

    void fill(double value) {
        fill(std::complex<double>(value, 0.0));
    }

    void set(uint64_t i, uint64_t j, std::complex<double> value) {
        ensure_unique();
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        const uint64_t storage_cols = base_cols();
        const uint64_t idx = is_transposed() ? (j * storage_cols + i) : (i * storage_cols + j);
        real_data()[idx] = pycauset::float16_t(value.real());
        imag_data()[idx] = pycauset::float16_t(value.imag());
    }

    std::complex<double> get(uint64_t i, uint64_t j) const {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        const uint64_t storage_cols = base_cols();
        const uint64_t idx = is_transposed() ? (j * storage_cols + i) : (i * storage_cols + j);
        return {
            static_cast<double>(real_data()[idx]),
            static_cast<double>(imag_data()[idx]),
        };
    }

    double element_as_double(uint64_t i, uint64_t j) const override {
        (void)i;
        (void)j;
        throw std::runtime_error("Complex matrix does not support get_element_as_double; use get_element_as_complex");
    }

    std::complex<double> element_as_complex(uint64_t i, uint64_t j) const override {
        const std::complex<double> v = get(i, j);
        std::complex<double> z = (scalar_ == 1.0) ? v : (v * scalar_);
        if (is_conjugated()) {
            z = std::conj(z);
        }
        return z;
    }

    void set_element_double(uint64_t i, uint64_t j, double value) override {
        set(i, j, std::complex<double>(value, 0.0));
    }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto out = std::make_unique<ComplexFloat16Matrix>(base_rows(), base_cols(), std::move(mapper));
        out->set_scalar(scalar_ * factor);
        out->set_seed(seed_);
        out->set_transposed(is_transposed());
        out->set_conjugated(is_conjugated());
        if (result_file.empty()) {
            out->set_temporary(true);
        }
        return out;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    std::unique_ptr<MatrixBase> multiply_scalar(std::complex<double> factor, const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto out = std::make_unique<ComplexFloat16Matrix>(base_rows(), base_cols(), std::move(mapper));
        out->set_scalar(scalar_ * (is_conjugated() ? std::conj(factor) : factor));  // conj(raw * scalar), F-071
        out->set_seed(seed_);
        out->set_transposed(is_transposed());
        out->set_conjugated(is_conjugated());
        if (result_file.empty()) {
            out->set_temporary(true);
        }
        return out;
    }



    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override {
        if (result_file.empty()) {
            auto out = std::make_unique<ComplexFloat16Matrix>(base_rows(), base_cols(), mapper_);
            out->set_transposed(!is_transposed());
            out->set_scalar(scalar_);
            out->set_seed(seed_);
            out->set_conjugated(is_conjugated());
            out->set_temporary(is_temporary());
            return out;
        }

        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto out = std::make_unique<ComplexFloat16Matrix>(base_rows(), base_cols(), std::move(mapper));
        out->set_transposed(!is_transposed());
        out->set_scalar(scalar_);
        out->set_seed(seed_);
        out->set_conjugated(is_conjugated());
        return out;
    }
};

} // namespace pycauset
