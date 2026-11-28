#pragma once
#include "pycauset/math/ViewValue.hpp"

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/vector/DenseVector.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pycauset {

template <typename T>
class DenseMatrix : public DenseMatrixBase {
public:
    using MatrixBase::operator=;

    DenseMatrix(uint64_t n, const std::string& backing_file = "")
        : DenseMatrix(n, n, backing_file) {}

    DenseMatrix(uint64_t rows, uint64_t cols, const std::string& backing_file = "")
        : DenseMatrixBase(rows, cols, pycauset::Structure::DENSE, MatrixTraits<T>::data_type) {
        const uint64_t size_in_bytes = rows * cols * sizeof(T);
        initialize_storage(size_in_bytes,
                           backing_file,
                           std::string("dense_") + MatrixTraits<T>::name,
                           sizeof(T),
                           pycauset::Structure::DENSE,
                           MatrixTraits<T>::data_type,
                           rows,
                           cols);
    }

    DenseMatrix(uint64_t n,
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed)
        : DenseMatrix(n, n, backing_file, offset, seed, scalar, is_transposed) {}

    DenseMatrix(uint64_t rows,
                uint64_t cols,
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed)
        : DenseMatrixBase(rows, cols, pycauset::Structure::DENSE, MatrixTraits<T>::data_type) {
        const uint64_t size_in_bytes = rows * cols * sizeof(T);
        initialize_storage(size_in_bytes,
                           backing_file,
                           "",
                           sizeof(T),
                           pycauset::Structure::DENSE,
                           MatrixTraits<T>::data_type,
                           rows,
                           cols,
                           offset,
                           false);

        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    DenseMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : DenseMatrix(n, n, std::move(mapper)) {}

    DenseMatrix(uint64_t rows, uint64_t cols, std::shared_ptr<MemoryMapper> mapper)
        : DenseMatrixBase(rows, cols, std::move(mapper), pycauset::Structure::DENSE, MatrixTraits<T>::data_type) {}

    DenseMatrix(uint64_t view_rows,
                uint64_t view_cols,
                std::shared_ptr<MemoryMapper> mapper,
                uint64_t base_rows,
                uint64_t base_cols,
                uint64_t row_offset,
                uint64_t col_offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed,
                bool is_conjugated)
        : DenseMatrixBase(base_rows, base_cols, std::move(mapper), pycauset::Structure::DENSE, MatrixTraits<T>::data_type, seed, scalar, is_transposed) {
        set_view(view_rows, view_cols, row_offset, col_offset);
        set_conjugated(is_conjugated);
    }

    void set(uint64_t i, uint64_t j, T value) {
        ensure_unique();
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        const uint64_t storage_cols = base_cols();
        const uint64_t storage_row = row_offset() + (is_transposed() ? j : i);
        const uint64_t storage_col = col_offset() + (is_transposed() ? i : j);
        const uint64_t idx = storage_row * storage_cols + storage_col;
        data()[idx] = value;
    }

    void set_element_double(uint64_t i, uint64_t j, double value) override {
        set(i, j, static_cast<T>(value));
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        const uint64_t storage_cols = base_cols();
        const uint64_t storage_row = row_offset() + (is_transposed() ? j : i);
        const uint64_t storage_col = col_offset() + (is_transposed() ? i : j);
        const uint64_t idx = storage_row * storage_cols + storage_col;
        return data()[idx];
    }

    // Every visible element becomes `value`: the view's factor and conjugation are dropped
        // (they applied to the old values; a scaled view kept the factor and read 2x the fill,
        // review 3.2 C-5).
    void fill(T value) {
        ensure_unique();
        set_scalar(1.0);
        set_conjugated(false);
        const bool full_span = row_offset() == 0 && col_offset() == 0 && !is_transposed() && rows() == base_rows() && cols() == base_cols();
        if (full_span) {
            T* ptr = data();
            const uint64_t total = base_rows() * base_cols();
            pycauset::ParallelFor(0, total, [&](size_t i) { ptr[i] = value; });
            return;
        }

        for (uint64_t i = 0; i < rows(); ++i) {
            for (uint64_t j = 0; j < cols(); ++j) {
                set(i, j, value);
            }
        }
    }

    void set_identity() {
        // The visible matrix (a window too): it looped over the storage's rows with logical
        // indices and threw after zeroing a window.
        if (rows() != cols()) {
            throw std::invalid_argument("Identity requires a square matrix");
        }
        fill(pycauset::scalar::from_double<T>(0.0));
        for (uint64_t i = 0; i < rows(); ++i) {
            set(i, i, pycauset::scalar::from_double<T>(1.0));
        }
    }

    double element_as_double(uint64_t i, uint64_t j) const override {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            throw std::runtime_error("Complex matrix does not support get_element_as_double; use get_element_as_complex");
        }
        // The one view rule (integer views wrap like NumPy; review 3.2 C-9: a complex multiply
        // here turned (inf) * 2 into inf + nan j).
        return pycauset::view_value(get(i, j), scalar_, false).real();
    }

    std::complex<double> element_as_complex(uint64_t i, uint64_t j) const override {
        return pycauset::view_value(get(i, j), scalar_, is_conjugated());
    }

    T* data() { return static_cast<T*>(require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(require_mapper()->get_data()); }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        (void)result_file;
        return scaled_clone(factor);
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    std::unique_ptr<MatrixBase> multiply_scalar(std::complex<double> factor, const std::string& result_file = "") const override {
        (void)result_file;
        return scaled_clone(factor);
    }



    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override {
        (void)result_file;
        return transposed_clone();
    }

};

} // namespace pycauset
