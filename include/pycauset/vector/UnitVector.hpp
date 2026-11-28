#pragma once

#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/math/ViewValue.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>

namespace pycauset {

class UnitVector : public VectorBase {
public:
    // The standard basis vector e_k of any dtype (no element storage; read-only).
    UnitVector(uint64_t n, uint64_t active_index, const std::string& backing_file = "",
               DataType dtype = DataType::FLOAT64)
        : VectorBase(n, 0, backing_file, "unit_vector", pycauset::Structure::UNIT_VECTOR, dtype),
          active_index_(active_index) {
        if (active_index >= n) {
            throw std::out_of_range("Active index out of bounds");
        }
        set_seed(active_index);  // the active index is stored in the seed field for persistence
    }

    // Constructor for loading
    UnitVector(uint64_t n, uint64_t active_index, const std::string& backing_file, size_t offset, uint64_t seed,
               std::complex<double> scalar, bool is_transposed, DataType dtype = DataType::FLOAT64)
        : VectorBase(n, 0, backing_file, offset, seed, scalar, is_transposed, pycauset::Structure::UNIT_VECTOR, dtype),
          active_index_(active_index) {}

    UnitVector(uint64_t n, std::shared_ptr<MemoryMapper> mapper, DataType dtype = DataType::FLOAT64)
        : VectorBase(n, std::move(mapper), pycauset::Structure::UNIT_VECTOR, dtype) {
        active_index_ = get_seed();  // recover the active index from the seed
    }

    // The active element's value: 1 in the dtype, through the view rule (math/ViewValue.hpp).
    std::complex<double> active_value() const {
        const std::complex<double> s = scalar_;
        const bool c = is_conjugated();
        switch (data_type_) {
            case DataType::BIT: return view_value(true, s, c);
            case DataType::INT8: return view_value(int8_t{1}, s, c);
            case DataType::INT16: return view_value(int16_t{1}, s, c);
            case DataType::INT32: return view_value(int32_t{1}, s, c);
            case DataType::INT64: return view_value(int64_t{1}, s, c);
            case DataType::UINT8: return view_value(uint8_t{1}, s, c);
            case DataType::UINT16: return view_value(uint16_t{1}, s, c);
            case DataType::UINT32: return view_value(uint32_t{1}, s, c);
            case DataType::UINT64: return view_value(uint64_t{1}, s, c);
            default: return view_value(1.0, s, c);  // floats and complex: 1 is exact everywhere
        }
    }

    double element_as_double(uint64_t i) const override {
        if (is_complex_dtype(data_type_)) {
            throw std::runtime_error("Complex vector does not support get_element_as_double; use get_element_as_complex");
        }
        return element_as_complex(i).real();
    }

    std::complex<double> element_as_complex(uint64_t i) const override {
        if (i >= n_) throw std::out_of_range("Index out of bounds");
        return i == active_index_ ? active_value() : std::complex<double>(0.0, 0.0);
    }

    std::unique_ptr<VectorBase> transpose(const std::string& saveas = "") const override {
        auto result = std::make_unique<UnitVector>(n_, active_index_, saveas, data_type_);
        result->set_scalar(scalar_);
        result->set_transposed(!is_transposed());
        result->set_conjugated(is_conjugated());
        return result;
    }

    uint64_t get_active_index() const { return active_index_; }

    // The generic VectorBase::clone rebuilds through the factory, which read the active index
    // before setting it: u.conj() moved the 1 to index 0 (F-077).
    std::unique_ptr<PersistentObject> clone() const override {
        auto out = std::make_unique<UnitVector>(n_, mapper_, data_type_);
        out->active_index_ = active_index_;
        out->set_seed(get_seed());
        out->set_scalar(get_scalar());
        out->set_transposed(is_transposed());
        out->set_conjugated(is_conjugated());
        return out;
    }

private:
    uint64_t active_index_;
};

} // namespace pycauset
