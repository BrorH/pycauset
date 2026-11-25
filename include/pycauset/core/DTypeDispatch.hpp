#pragma once
// One place that turns a runtime (Structure, DataType) pair into the concrete C++ type.
//
// Replaces the per-dtype dynamic_cast ladders that used to be copy-pasted through the
// bindings. Usage:
//
//     pycauset::visit_matrix(m, [&](auto& typed) { ... typed.get(i, j) ... });
//
// `typed` is the concrete object (DenseMatrix<int8_t>&, TriangularMatrix<bool>&,
// ComplexFloat16Matrix&, ...). Which (structure, dtype) pairs exist is decided once, by the
// catalogue below (`offered`, `ClassOf`); pairs it does not offer are never instantiated.

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "pycauset/core/Float16.hpp"
#include "pycauset/core/Types.hpp"
#include "pycauset/matrix/BitClosureMatrix.hpp"
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
        case Structure::HERMITIAN: return "hermitian";
        case Structure::SKEW_HERMITIAN: return "skew_hermitian";
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
    if (s == "hermitian") return Structure::HERMITIAN;
    if (s == "skew_hermitian") return Structure::SKEW_HERMITIAN;
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
    // Say why: the catalogue's reasons (see `offered` below).
    std::string why;
    const bool cplx = dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 ||
                      dt == DataType::COMPLEX_FLOAT64;
    if (s == Structure::HERMITIAN || s == Structure::SKEW_HERMITIAN) {
        if (!cplx) why = ": Hermitian matrices need a complex dtype (for real dtypes use SymmetricMatrix / AntiSymmetricMatrix)";
    } else if (s == Structure::ANTISYMMETRIC) {
        if (dt == DataType::BIT || dt == DataType::UINT8 || dt == DataType::UINT16 || dt == DataType::UINT32 ||
            dt == DataType::UINT64)
            why = ": antisymmetric matrices need a signed dtype";
    }
    throw std::invalid_argument(std::string("no ") + structure_name(s) + " matrix of dtype " + dtype_name(dt) + why);
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

// ------------------------------------------------------------------ the catalogue
// Which (structure, element type) pairs exist, and which class holds each. This is the ONLY
// place that decides (review N.5.1): ObjectFactory, visit_matrix, loading (open_matrix) and
// the constructor bindings all go through it. To offer a new pair, change `offered` (and make
// the class template work for the element type); see protocols/Adding Types.md.
template <Structure S, typename T>
constexpr bool offered() {
    constexpr bool cplx = is_complex_element_v<T>;
    constexpr bool signed_like = !(std::is_unsigned_v<T> || std::is_same_v<T, bool>);
    if constexpr (S == Structure::DENSE) {
        return true;
    } else if constexpr (S == Structure::TRIANGULAR || S == Structure::DIAGONAL || S == Structure::IDENTITY ||
                         S == Structure::SYMMETRIC) {
        return true;
    } else if constexpr (S == Structure::ANTISYMMETRIC) {
        return signed_like;  // A(j,i) = -A(i,j) needs negation (F-065)
    } else if constexpr (S == Structure::HERMITIAN || S == Structure::SKEW_HERMITIAN) {
        return cplx;  // for real dtypes these are symmetric / antisymmetric
    } else {
        return false;
    }
}

// Element type of structured storage: complex_float16 is the interleaved complex_half there
// (dense complex_float16 keeps its split planes, ComplexFloat16Matrix).
template <typename T>
using structured_element_t = std::conditional_t<std::is_same_v<T, complex_float16_marker>, complex_half, T>;

template <Structure S, typename T> struct ClassOf { using type = void; };
template <typename T> struct ClassOf<Structure::DENSE, T> { using type = typename DenseMatrixOf<T>::type; };
template <typename T> struct ClassOf<Structure::TRIANGULAR, T> { using type = TriangularMatrix<structured_element_t<T>>; };
template <typename T> struct ClassOf<Structure::DIAGONAL, T> { using type = DiagonalMatrix<structured_element_t<T>>; };
template <typename T> struct ClassOf<Structure::IDENTITY, T> { using type = IdentityMatrix<structured_element_t<T>>; };
template <typename T> struct ClassOf<Structure::SYMMETRIC, T> { using type = SymmetricMatrix<structured_element_t<T>>; };
template <typename T> struct ClassOf<Structure::ANTISYMMETRIC, T> { using type = SymmetricMatrix<structured_element_t<T>>; };
template <typename T> struct ClassOf<Structure::HERMITIAN, T> { using type = SymmetricMatrix<structured_element_t<T>>; };
template <typename T> struct ClassOf<Structure::SKEW_HERMITIAN, T> { using type = SymmetricMatrix<structured_element_t<T>>; };

template <Structure S>
using structure_tag = std::integral_constant<Structure, S>;

// Calls f(structure_tag<S>{}) for a matrix structure.
template <typename F>
decltype(auto) visit_structure(Structure s, F&& f) {
    switch (s) {
        case Structure::DENSE: return f(structure_tag<Structure::DENSE>{});
        case Structure::TRIANGULAR: return f(structure_tag<Structure::TRIANGULAR>{});
        case Structure::DIAGONAL: return f(structure_tag<Structure::DIAGONAL>{});
        case Structure::IDENTITY: return f(structure_tag<Structure::IDENTITY>{});
        case Structure::SYMMETRIC: return f(structure_tag<Structure::SYMMETRIC>{});
        case Structure::ANTISYMMETRIC: return f(structure_tag<Structure::ANTISYMMETRIC>{});
        case Structure::HERMITIAN: return f(structure_tag<Structure::HERMITIAN>{});
        case Structure::SKEW_HERMITIAN: return f(structure_tag<Structure::SKEW_HERMITIAN>{});
        default: throw std::invalid_argument(std::string("not a matrix structure: ") + structure_name(s));
    }
}

// Is the (structure, dtype) pair offered (runtime form of `offered`)?
inline bool is_offered(Structure s, DataType dt) {
    if (s == Structure::VECTOR || s == Structure::UNIT_VECTOR || s == Structure::UNKNOWN) return false;
    return visit_structure(s, [&](auto st) {
        return visit_dtype(dt, [&](auto tag) { return offered<decltype(st)::value, typename decltype(tag)::type>(); });
    });
}

// Structure-specific construction options (one struct, so nothing is dropped on the way:
// a lost has_diagonal read shifted values, F-064 / F-076).
struct MatrixOptions {
    bool has_diagonal = false;  // TRIANGULAR only
};

namespace detail {
inline void require_square(Structure s, uint64_t rows, uint64_t cols) {
    if (s != Structure::DENSE && s != Structure::IDENTITY && rows != cols) {
        throw std::invalid_argument("Structure requires square dimensions");
    }
}

// One builder for the three constructor shapes every class has. Mode::Create: new storage
// (`file` may be empty); Mode::Open: existing storage at `offset` with a view; Mode::Adopt:
// share a mapper (clone).
enum class Build { Create, Open, Adopt };

struct BuildArgs {
    uint64_t rows = 0, cols = 0;
    std::string file;
    size_t offset = 0;
    uint64_t seed = 0;
    std::complex<double> scalar = 1.0;
    bool transposed = false;
    std::shared_ptr<MemoryMapper> mapper;
    MatrixOptions opts;
};

template <Build B>
std::unique_ptr<MatrixBase> build_matrix(Structure s, DataType dt, const BuildArgs& a) {
    require_square(s, a.rows, a.cols);
    return visit_structure(s, [&](auto st) -> std::unique_ptr<MatrixBase> {
        constexpr Structure S = decltype(st)::value;
        return visit_dtype(dt, [&](auto tag) -> std::unique_ptr<MatrixBase> {
            using T = typename decltype(tag)::type;
            if constexpr (!offered<S, T>()) {
                throw_unsupported(S, dt);
            } else {
                using C = typename ClassOf<S, T>::type;
                const uint64_t n = a.rows;
                if constexpr (S == Structure::DENSE || S == Structure::IDENTITY) {
                    if constexpr (B == Build::Create) return std::make_unique<C>(a.rows, a.cols, a.file);
                    else if constexpr (B == Build::Open)
                        return std::make_unique<C>(a.rows, a.cols, a.file, a.offset, a.seed, a.scalar, a.transposed);
                    else return std::make_unique<C>(a.rows, a.cols, a.mapper);
                } else if constexpr (S == Structure::TRIANGULAR && std::is_same_v<T, bool>) {
                    // Bit triangles are strictly upper triangular (with-diagonal: review N.5.6).
                    if (a.opts.has_diagonal) throw std::invalid_argument("bit triangular matrices are strictly upper triangular");
                    if constexpr (B == Build::Create) return std::make_unique<C>(n, a.file);
                    else if constexpr (B == Build::Open)
                        return std::make_unique<C>(n, a.file, a.offset, a.seed, a.scalar, a.transposed);
                    else return std::make_unique<C>(n, a.mapper);
                } else if constexpr (S == Structure::TRIANGULAR) {
                    const bool d = a.opts.has_diagonal;
                    if constexpr (B == Build::Create) return std::make_unique<C>(n, a.file, d);
                    else if constexpr (B == Build::Open)
                        return std::make_unique<C>(n, a.file, a.offset, a.seed, a.scalar, a.transposed, d);
                    else return std::make_unique<C>(n, a.mapper, d);
                } else if constexpr (S == Structure::DIAGONAL) {
                    if constexpr (B == Build::Create) return std::make_unique<C>(n, a.file);
                    else if constexpr (B == Build::Open)
                        return std::make_unique<C>(n, a.file, a.offset, a.seed, a.scalar, a.transposed);
                    else return std::make_unique<C>(n, a.mapper);
                } else {  // the symmetric family: the structure is the kind
                    if constexpr (B == Build::Create) return std::make_unique<C>(n, a.file, S);
                    else if constexpr (B == Build::Open)
                        return std::make_unique<C>(n, a.file, a.offset, a.seed, a.scalar, a.transposed, S);
                    else return std::make_unique<C>(n, a.mapper, S);
                }
            }
        });
    });
}
}  // namespace detail

// New storage for a (structure, dtype) pair.
inline std::unique_ptr<MatrixBase> make_matrix(Structure s, DataType dt, uint64_t rows, uint64_t cols,
                                               const std::string& file = "", MatrixOptions opts = {}) {
    detail::BuildArgs a;
    a.rows = rows; a.cols = cols; a.file = file; a.opts = opts;
    return detail::build_matrix<detail::Build::Create>(s, dt, a);
}

// Existing storage (persistence): `rows` x `cols` is the storage shape, the view is applied.
inline std::unique_ptr<MatrixBase> open_matrix(Structure s, DataType dt, uint64_t rows, uint64_t cols,
                                               const std::string& file, size_t offset, uint64_t seed,
                                               std::complex<double> scalar, bool transposed, MatrixOptions opts = {}) {
    detail::BuildArgs a;
    a.rows = rows; a.cols = cols; a.file = file; a.offset = offset; a.seed = seed; a.scalar = scalar;
    a.transposed = transposed; a.opts = opts;
    return detail::build_matrix<detail::Build::Open>(s, dt, a);
}

// A new object sharing `mapper` (clone); the caller sets the view.
inline std::unique_ptr<MatrixBase> adopt_matrix(Structure s, DataType dt, std::shared_ptr<MemoryMapper> mapper,
                                                uint64_t rows, uint64_t cols, MatrixOptions opts = {}) {
    detail::BuildArgs a;
    a.rows = rows; a.cols = cols; a.mapper = std::move(mapper); a.opts = opts;
    return detail::build_matrix<detail::Build::Adopt>(s, dt, a);
}

// Calls f(concrete&) for a matrix. Works for const and non-const MatrixBase.
// Every branch must return the same type (use void or py::object in callers).
template <typename M, typename F>
decltype(auto) visit_matrix(M& m, F&& f) {
    static_assert(std::is_base_of_v<MatrixBase, std::remove_const_t<M>>);
    const Structure s = m.get_structure();
    const DataType dt = m.get_data_type();
    // Closure views (LAZY.md L2b) are symmetric/antisymmetric int32 with bit storage.
    if (s == Structure::SYMMETRIC || s == Structure::ANTISYMMETRIC) {
        using CT = std::conditional_t<std::is_const_v<M>, const BitClosureMatrix, BitClosureMatrix>;
        if (auto* c = dynamic_cast<CT*>(&m)) return f(*c);
    }
    return visit_dtype(dt, [&](auto tag) -> decltype(auto) {
        using T = typename decltype(tag)::type;
#define PYCAUSET_VISIT_CASE(S)                                                                \
        case S:                                                                               \
            if constexpr (offered<S, T>()) return detail::call_as<typename ClassOf<S, T>::type>(m, f); \
            break;
        switch (s) {
            case Structure::DENSE: return detail::call_as<typename ClassOf<Structure::DENSE, T>::type>(m, f);
            PYCAUSET_VISIT_CASE(Structure::TRIANGULAR)
            PYCAUSET_VISIT_CASE(Structure::DIAGONAL)
            PYCAUSET_VISIT_CASE(Structure::IDENTITY)
            PYCAUSET_VISIT_CASE(Structure::SYMMETRIC)
            PYCAUSET_VISIT_CASE(Structure::ANTISYMMETRIC)
            PYCAUSET_VISIT_CASE(Structure::HERMITIAN)
            PYCAUSET_VISIT_CASE(Structure::SKEW_HERMITIAN)
            default: break;
        }
#undef PYCAUSET_VISIT_CASE
        throw_unsupported(s, dt);
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
