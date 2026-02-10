#pragma once

#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/StorageUtils.hpp"

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace pycauset {

class ComplexFloat16Vector : public VectorBase {
public:
    ComplexFloat16Vector(uint64_t n, const std::string& backing_file = "")
        : VectorBase(n) {
        const uint64_t size_in_bytes = 2ULL * n * sizeof(pycauset::float16_t);
        initialize_storage(
            size_in_bytes,
            backing_file,
            std::string("vector_complex_float16"),
            sizeof(pycauset::float16_t),
            pycauset::MatrixType::DENSE_FLOAT,
            pycauset::DataType::COMPLEX_FLOAT16,
            n,
            1);
    }

    ComplexFloat16Vector(uint64_t n,
                         const std::string& backing_file,
                         size_t offset,
                         uint64_t seed,
                         std::complex<double> scalar,
                         bool is_transposed)
        : VectorBase(n) {
        const uint64_t size_in_bytes = 2ULL * n * sizeof(pycauset::float16_t);
        initialize_storage(
            size_in_bytes,
            backing_file,
            "",
            sizeof(pycauset::float16_t),
            pycauset::MatrixType::DENSE_FLOAT,
            pycauset::DataType::COMPLEX_FLOAT16,
            n,
            1,
            offset,
            false);

        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    ComplexFloat16Vector(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : VectorBase(n, std::move(mapper), pycauset::MatrixType::DENSE_FLOAT, pycauset::DataType::COMPLEX_FLOAT16) {}

    ~ComplexFloat16Vector() override = default;

    pycauset::float16_t* real_data() { return static_cast<pycauset::float16_t*>(require_mapper()->get_data()); }
    const pycauset::float16_t* real_data() const {
        return static_cast<const pycauset::float16_t*>(require_mapper()->get_data());
    }

    pycauset::float16_t* imag_data() { return real_data() + n_; }
    const pycauset::float16_t* imag_data() const { return real_data() + n_; }

    void set(uint64_t i, std::complex<double> value) {
        ensure_unique();
        if (i >= n_) throw std::out_of_range("Index out of bounds");
        real_data()[i] = pycauset::float16_t(value.real());
        imag_data()[i] = pycauset::float16_t(value.imag());
    }

    std::complex<double> get(uint64_t i) const {
        if (i >= n_) throw std::out_of_range("Index out of bounds");
        return {static_cast<double>(real_data()[i]), static_cast<double>(imag_data()[i])};
    }

    double get_element_as_double(uint64_t i) const override {
        (void)i;
        throw std::runtime_error("Complex vector does not support get_element_as_double; use get_element_as_complex");
    }

    std::complex<double> get_element_as_complex(uint64_t i) const override {
        const std::complex<double> v = get(i);
        std::complex<double> z = (scalar_ == 1.0) ? v : (v * scalar_);
        if (is_conjugated()) {
            z = std::conj(z);
        }
        return z;
    }

    std::unique_ptr<VectorBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto out = std::make_unique<ComplexFloat16Vector>(n_, std::move(mapper));
        out->set_scalar(scalar_ * factor);
        out->set_seed(seed_);
        out->set_transposed(is_transposed());
        out->set_conjugated(is_conjugated());
        if (result_file.empty()) {
            out->set_temporary(true);
        }
        return out;
    }

    std::unique_ptr<VectorBase> multiply_scalar(std::complex<double> factor, const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto out = std::make_unique<ComplexFloat16Vector>(n_, std::move(mapper));
        out->set_scalar(scalar_ * factor);
        out->set_seed(seed_);
        out->set_transposed(is_transposed());
        out->set_conjugated(is_conjugated());
        if (result_file.empty()) {
            out->set_temporary(true);
        }
        return out;
    }

    std::unique_ptr<VectorBase> add_scalar(double scalar, const std::string& result_file = "") const override {
        auto result = std::make_unique<ComplexFloat16Vector>(n_, result_file);

        const auto* rsrc = real_data();
        const auto* isrc = imag_data();
        auto* rdst = result->real_data();
        auto* idst = result->imag_data();

        const std::complex<double> add = {scalar, 0.0};
        for (uint64_t i = 0; i < n_; ++i) {
            const std::complex<double> v = {static_cast<double>(rsrc[i]), static_cast<double>(isrc[i])};
            const std::complex<double> out = v * scalar_ + add;
            rdst[i] = pycauset::float16_t(out.real());
            idst[i] = pycauset::float16_t(out.imag());
        }

        result->set_scalar(1.0);
        result->set_seed(seed_);
        result->set_transposed(is_transposed());
        if (result_file.empty()) {
            result->set_temporary(true);
        }
        return result;
    }

    std::unique_ptr<VectorBase> transpose(const std::string& result_file = "") const override {
        if (result_file.empty()) {
            auto out = std::make_unique<ComplexFloat16Vector>(n_, mapper_);
            out->set_transposed(!is_transposed());
            out->set_scalar(scalar_);
            out->set_seed(seed_);
            out->set_conjugated(is_conjugated());
            out->set_temporary(is_temporary());
            return out;
        }

        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto out = std::make_unique<ComplexFloat16Vector>(n_, std::move(mapper));
        out->set_transposed(!is_transposed());
        out->set_scalar(scalar_);
        out->set_seed(seed_);
        out->set_conjugated(is_conjugated());
        return out;
    }
};

} // namespace pycauset
