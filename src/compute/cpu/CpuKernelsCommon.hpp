// Kernel helpers shared by the CPU solver files (split from CpuSolver.cpp, review 4.2).
// Internal to src/compute/cpu: include after the solver headers, not part of the public API.
#pragma once

namespace pycauset::cpu_detail {

    template <typename T>
    constexpr bool is_std_complex_v = false;

    template <>
    constexpr bool is_std_complex_v<std::complex<float>> = true;

    template <>
    constexpr bool is_std_complex_v<std::complex<double>> = true;

    inline bool is_broadcast_compatible_dim(uint64_t in_dim, uint64_t out_dim) {
        return (in_dim == out_dim) || (in_dim == 1);
    }

    // Reads element (i, j) of a dense operand as the result type T, honouring the view
    // (transpose, offsets, broadcasting, scale factor, conjugation). Built once per operation.
    template <typename T>
    struct StridedReader {
        using RowFn = void (*)(const StridedReader&, uint64_t, uint64_t, uint64_t, T*);
        const void* data = nullptr;
        int64_t base = 0, si = 0, sj = 0;
        std::complex<double> scalar = 1.0;
        int64_t kf = 1;  // integer factor of an integer view (applied with wrapping in U)
        bool scaled = false, conj = false;
        RowFn row_fn = nullptr;
        // Fills out[0 .. j1-j0) with elements (i, j0) .. (i, j1-1).
        void row(uint64_t i, uint64_t j0, uint64_t j1, T* out) const { row_fn(*this, i, j0, j1, out); }
    };

    template <typename T, typename U>
    void strided_row(const StridedReader<T>& r, uint64_t i, uint64_t j0, uint64_t j1, T* out) {
        const U* src = static_cast<const U*>(r.data) + r.base + static_cast<int64_t>(i) * r.si;
        const int64_t sj = r.sj;
        const uint64_t n = j1 - j0;
        src += static_cast<int64_t>(j0) * sj;
        if constexpr (std::is_integral_v<U> && !std::is_same_v<U, bool>) {
            // Integer operand: the view's value is wrap(x * k) in U, then converted to T.
            if (r.scaled) {
                const int64_t k = r.kf;
                for (uint64_t q = 0; q < n; ++q) {
                    const U v = pycauset::scalar::wrap_scale(src[static_cast<int64_t>(q) * sj], k);
                    if constexpr (is_std_complex_v<T>) out[q] = static_cast<T>(std::complex<double>(static_cast<double>(v), 0.0));
                    else if constexpr (std::is_integral_v<T>) out[q] = static_cast<T>(v);
                    else out[q] = static_cast<T>(static_cast<double>(v));
                }
                return;
            }
        }
        if constexpr (is_std_complex_v<T>) {
            // The shared view rule (math/ViewValue.hpp): conj(raw * s), not conj(raw) * s (F-086).
            const bool conj = r.conj;
            const std::complex<double> s = r.scaled ? r.scalar : std::complex<double>(1.0, 0.0);
            for (uint64_t q = 0; q < n; ++q) {
                out[q] = static_cast<T>(pycauset::view_value(src[static_cast<int64_t>(q) * sj], s, conj));
            }
        } else if constexpr (std::is_integral_v<T>) {
            for (uint64_t q = 0; q < n; ++q) out[q] = static_cast<T>(src[static_cast<int64_t>(q) * sj]);
        } else {
            if (r.scaled) {
                const double s = r.scalar.real();
                for (uint64_t q = 0; q < n; ++q) out[q] = static_cast<T>(static_cast<double>(src[static_cast<int64_t>(q) * sj]) * s);
            } else if constexpr (std::is_same_v<T, U>) {
                if (sj == 1) {
                    std::memcpy(out, src, n * sizeof(T));
                } else {
                    for (uint64_t q = 0; q < n; ++q) out[q] = src[static_cast<int64_t>(q) * sj];
                }
            } else {
                for (uint64_t q = 0; q < n; ++q) out[q] = static_cast<T>(static_cast<double>(src[static_cast<int64_t>(q) * sj]));
            }
        }
    }

    // Which (result T, operand U) pairs the strided path handles: any operand into a complex
    // result; real operands into a real float result; integer operands into an integer result.
    template <typename T, typename U>
    constexpr bool strided_supported_v =
        is_std_complex_v<T> || (!is_std_complex_v<U> && (!std::is_integral_v<T> || std::is_integral_v<U>));

    template <typename T, typename U>
    bool try_strided(const MatrixBase& m, uint64_t rows, uint64_t cols, StridedReader<T>& r) {
        if constexpr (!strided_supported_v<T, U>) {
            return false;
        } else {
            const auto* d = dynamic_cast<const DenseMatrix<U>*>(&m);
            if (d == nullptr) return false;
            if constexpr (std::is_integral_v<U> && !std::is_same_v<U, bool>) {
                // Integer operands carry only exact integer factors (integer views, F-042).
                if (!pycauset::scalar::is_exact_int_factor(m.get_scalar())) return false;
                r.kf = static_cast<int64_t>(m.get_scalar().real());
            }
            if constexpr (!is_std_complex_v<T>) {
                if (m.get_scalar().imag() != 0.0) return false;
            }
            const int64_t bc = static_cast<int64_t>(d->base_cols());
            const bool tr = d->is_transposed();
            int64_t si = tr ? 1 : bc, sj = tr ? bc : 1;
            if (d->rows() == 1 && rows != 1) si = 0;  // broadcasting
            if (d->cols() == 1 && cols != 1) sj = 0;
            r.data = d->data();
            r.base = static_cast<int64_t>(d->row_offset()) * bc + static_cast<int64_t>(d->col_offset());
            r.si = si;
            r.sj = sj;
            r.scalar = m.get_scalar();
            r.scaled = r.scalar != std::complex<double>(1.0, 0.0);
            r.conj = m.is_conjugated();
            r.row_fn = &strided_row<T, U>;
            return true;
        }
    }

    template <typename T>
    bool make_strided_reader(const MatrixBase& m, uint64_t rows, uint64_t cols, StridedReader<T>& r) {
        if (m.get_structure() != Structure::DENSE) return false;
        switch (m.get_data_type()) {
            case DataType::INT8: return try_strided<T, int8_t>(m, rows, cols, r);
            case DataType::INT16: return try_strided<T, int16_t>(m, rows, cols, r);
            case DataType::INT32: return try_strided<T, int32_t>(m, rows, cols, r);
            case DataType::INT64: return try_strided<T, int64_t>(m, rows, cols, r);
            case DataType::UINT8: return try_strided<T, uint8_t>(m, rows, cols, r);
            case DataType::UINT16: return try_strided<T, uint16_t>(m, rows, cols, r);
            case DataType::UINT32: return try_strided<T, uint32_t>(m, rows, cols, r);
            case DataType::UINT64: return try_strided<T, uint64_t>(m, rows, cols, r);
            case DataType::FLOAT32: return try_strided<T, float>(m, rows, cols, r);
            case DataType::FLOAT64: return try_strided<T, double>(m, rows, cols, r);
            case DataType::COMPLEX_FLOAT32: return try_strided<T, std::complex<float>>(m, rows, cols, r);
            case DataType::COMPLEX_FLOAT64: return try_strided<T, std::complex<double>>(m, rows, cols, r);
            default: return false;  // bit, float16, complex16: per-element path
        }
    }

}  // namespace pycauset::cpu_detail
