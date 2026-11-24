#pragma once
// One place that turns a runtime (Structure, DataType) pair into the concrete C++ type.
//
// Replaces the per-dtype dynamic_cast ladders that used to be copy-pasted through the
// bindings. Usage:
//
//     pycauset::visit_matrix(m, [&](auto& typed) { ... typed.get(i, j) ... });
//
// `typed` is the concrete object (DenseMatrix<int8_t>&, TriangularMatrix<bool>&,
// ComplexFloat16Matrix&, ...). Combinations the engine never instantiates (e.g. a
// triangular complex matrix) are never named, so they don't need to compile.

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "pycauset/core/Float16.hpp"
#include "pycauset/core/Types.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/UnitVector.hpp"

namespace pycauset {

// Element-type marker for complex_float16, whose storage (split real/imag float16 planes)
// has its own classes rather than a DenseMatrix<T> instantiation.
struct complex_float16_marker {};

template <typename T>
struct type_tag {
    using type = T;
};

// Canonical dtype names (match python/pycauset/_internal/dtypes.py).
inline const char* dtype_name(DataType dt) {
    switch (dt) {
        case DataType::BIT: return "bool";
        case DataType::INT8: return "int8";
        case DataType::INT16: return "int16";
        case DataType::INT32: return "int32";
        case DataType::INT64: return "int64";
        case DataType::UINT8: return "uint8";
        case DataType::UINT16: return "uint16";
        case DataType::UINT32: return "uint32";
        case DataType::UINT64: return "uint64";
        case DataType::FLOAT16: return "float16";
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT64: return "float64";
        case DataType::COMPLEX_FLOAT16: return "complex_float16";
        case DataType::COMPLEX_FLOAT32: return "complex_float32";
        case DataType::COMPLEX_FLOAT64: return "complex_float64";
        default: return "unknown";
    }
}

inline const char* structure_name(Structure s) {
    switch (s) {
        case Structure::DENSE: return "dense";
        case Structure::TRIANGULAR: return "triangular";
        case Structure::DIAGONAL: return "diagonal";
        case Structure::IDENTITY: return "identity";
        case Structure::SYMMETRIC: return "symmetric";
        case Structure::ANTISYMMETRIC: return "antisymmetric";
        case Structure::VECTOR: return "vector";
        case Structure::UNIT_VECTOR: return "unit_vector";
        default: return "unknown";
    }
}

inline Structure parse_structure(const std::string& s) {
    if (s == "dense") return Structure::DENSE;
    if (s == "triangular") return Structure::TRIANGULAR;
    if (s == "diagonal") return Structure::DIAGONAL;
    if (s == "identity") return Structure::IDENTITY;
    if (s == "symmetric") return Structure::SYMMETRIC;
    if (s == "antisymmetric") return Structure::ANTISYMMETRIC;
    if (s == "vector") return Structure::VECTOR;
    if (s == "unit_vector") return Structure::UNIT_VECTOR;
    throw std::invalid_argument("unknown structure: " + s);
}

inline DataType parse_dtype_name(const std::string& s) {
    for (DataType dt : {DataType::BIT, DataType::INT8, DataType::INT16, DataType::INT32, DataType::INT64,
                        DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64,
                        DataType::FLOAT16, DataType::FLOAT32, DataType::FLOAT64,
                        DataType::COMPLEX_FLOAT16, DataType::COMPLEX_FLOAT32, DataType::COMPLEX_FLOAT64}) {
        if (s == dtype_name(dt)) return dt;
    }
    throw std::invalid_argument("unknown dtype: " + s);
}

// Calls f(type_tag<T>{}) with T the element type of `dt`.
template <typename F>
decltype(auto) visit_dtype(DataType dt, F&& f) {
    switch (dt) {
        case DataType::BIT: return f(type_tag<bool>{});
        case DataType::INT8: return f(type_tag<int8_t>{});
        case DataType::INT16: return f(type_tag<int16_t>{});
        case DataType::INT32: return f(type_tag<int32_t>{});
        case DataType::INT64: return f(type_tag<int64_t>{});
        case DataType::UINT8: return f(type_tag<uint8_t>{});
        case DataType::UINT16: return f(type_tag<uint16_t>{});
        case DataType::UINT32: return f(type_tag<uint32_t>{});
        case DataType::UINT64: return f(type_tag<uint64_t>{});
        case DataType::FLOAT16: return f(type_tag<float16_t>{});
        case DataType::FLOAT32: return f(type_tag<float>{});
        case DataType::FLOAT64: return f(type_tag<double>{});
        case DataType::COMPLEX_FLOAT16: return f(type_tag<complex_float16_marker>{});
        case DataType::COMPLEX_FLOAT32: return f(type_tag<std::complex<float>>{});
        case DataType::COMPLEX_FLOAT64: return f(type_tag<std::complex<double>>{});
        default: throw std::invalid_argument("unsupported dtype");
    }
}

template <typename T>
inline constexpr bool is_complex_element_v =
    std::is_same_v<T, complex_float16_marker> || std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

// Concrete classes per structure and element type.
template <typename T> struct DenseMatrixOf { using type = DenseMatrix<T>; };
template <> struct DenseMatrixOf<complex_float16_marker> { using type = ComplexFloat16Matrix; };
template <typename T> struct DenseVectorOf { using type = DenseVector<T>; };
template <> struct DenseVectorOf<complex_float16_marker> { using type = ComplexFloat16Vector; };

[[noreturn]] inline void throw_unsupported(Structure s, DataType dt) {
    throw std::invalid_argument(std::string("no ") + structure_name(s) + " matrix of dtype " + dtype_name(dt));
}

namespace detail {
template <typename Concrete, typename Base, typename F>
decltype(auto) call_as(Base& obj, F&& f) {
    using Target = std::conditional_t<std::is_const_v<Base>, const Concrete, Concrete>;
    auto* typed = dynamic_cast<Target*>(&obj);
    if (typed == nullptr) {
        throw std::logic_error("object does not match its (structure, dtype) tag");
    }
    return f(*typed);
}
}  // namespace detail

// Calls f(concrete&) for a matrix. Works for const and non-const MatrixBase.
// Every branch must return the same type (use void or py::object in callers).
template <typename M, typename F>
decltype(auto) visit_matrix(M& m, F&& f) {
    static_assert(std::is_base_of_v<MatrixBase, std::remove_const_t<M>>);
    const Structure s = m.get_structure();
    const DataType dt = m.get_data_type();
    return visit_dtype(dt, [&](auto tag) -> decltype(auto) {
        using T = typename decltype(tag)::type;
        if (s == Structure::DENSE) {
            return detail::call_as<typename DenseMatrixOf<T>::type>(m, f);
        }
        if constexpr (is_complex_element_v<T>) {
            throw_unsupported(s, dt);
        } else {
            switch (s) {
                case Structure::TRIANGULAR: return detail::call_as<TriangularMatrix<T>>(m, f);
                case Structure::SYMMETRIC:
                case Structure::ANTISYMMETRIC: return detail::call_as<SymmetricMatrix<T>>(m, f);
                case Structure::DIAGONAL:
                    if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, float>) throw_unsupported(s, dt);
                    else return detail::call_as<DiagonalMatrix<T>>(m, f);
                case Structure::IDENTITY:
                    if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, float>) throw_unsupported(s, dt);
                    else return detail::call_as<IdentityMatrix<T>>(m, f);
                default: throw_unsupported(s, dt);
            }
        }
    });
}

// Calls f(concrete&) for a vector.
template <typename V, typename F>
decltype(auto) visit_vector(V& v, F&& f) {
    static_assert(std::is_base_of_v<VectorBase, std::remove_const_t<V>>);
    if (v.get_structure() == Structure::UNIT_VECTOR) {
        return detail::call_as<UnitVector>(v, f);
    }
    return visit_dtype(v.get_data_type(), [&](auto tag) -> decltype(auto) {
        using T = typename decltype(tag)::type;
        return detail::call_as<typename DenseVectorOf<T>::type>(v, f);
    });
}

}  // namespace pycauset
