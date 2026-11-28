#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/math/VectorOps.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/math/Bitwise.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include <cmath>
#include <limits>
#include "bindings_common.hpp"
#include "element_conversion.hpp"
#include "scalar_args.hpp"
#include "pycauset/math/Reductions.hpp"
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <iostream>

#include "binding_warnings.hpp"

#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/core/ObjectFactory.hpp"

#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/VectorBase.hpp"

#include "numpy_export_guard.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/ParallelUtils.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "numpy_import.hpp"
#include "matrix_indexing.hpp"

using namespace pycauset;
using namespace pycauset_bindings::numpy_import;

namespace {

inline void maybe_warn_integer_matmul_overflow_risk_preflight(const MatrixBase& a, const MatrixBase& b) {
    // ADVISORY ONLY (policy: guesses may inform, never decide). Samples a fixed,
    // deterministic set of entries to estimate max|A|, max|B| and warns early when the
    // result may overflow int32. It never changes the result: the matmul kernel uses an
    // int64 accumulator and raises OverflowError exactly when a value doesn't fit.
    // Value bounds (review/features/VALUE_BOUNDS.md) will make this exact when known.
    auto* ai = dynamic_cast<const DenseMatrix<int32_t>*>(&a);
    auto* bi = dynamic_cast<const DenseMatrix<int32_t>*>(&b);
    if (!ai || !bi) {
        return;
    }

    const uint64_t m = a.rows();
    const uint64_t k = a.cols();
    const uint64_t n = b.cols();
    if (m < 128 || k < 128 || n < 128) {
        return;
    }

    const uint64_t sample_count = std::min<uint64_t>(std::min<uint64_t>(std::min<uint64_t>(m, k), n), 64);
    int64_t max_abs_a = 0;
    int64_t max_abs_b = 0;

    auto update_max_abs = [](int64_t& dst, int32_t v) {
        int64_t av = static_cast<int64_t>(v);
        if (av < 0) av = -av;
        if (av > dst) dst = av;
    };

    // Always include a few fixed positions for determinism.
    update_max_abs(max_abs_a, ai->get(0, 0));
    update_max_abs(max_abs_b, bi->get(0, 0));
    update_max_abs(max_abs_a, ai->get(m - 1, k - 1));
    update_max_abs(max_abs_b, bi->get(k - 1, n - 1));
    update_max_abs(max_abs_a, ai->get(0, k - 1));
    update_max_abs(max_abs_b, bi->get(k - 1, 0));
    update_max_abs(max_abs_a, ai->get(m - 1, 0));
    update_max_abs(max_abs_b, bi->get(0, n - 1));

    // Stride-based sample (no RNG; stable across runs).
    // Uses Knuth multiplicative hashing to spread indices.
    for (uint64_t s = 0; s < sample_count; ++s) {
        uint64_t i = (s * 2654435761u) % m;
        uint64_t j = (s * 805459861u + 1u) % k;
        uint64_t l = (s * 11400714819323198485ull + 2u) % n;
        update_max_abs(max_abs_a, ai->get(i, j));
        update_max_abs(max_abs_b, bi->get(j, l));
    }

    if (max_abs_a == 0 || max_abs_b == 0) {
        return;
    }

    const double scalar_factor = std::abs(a.get_scalar()) * std::abs(b.get_scalar());
    const double bound = static_cast<double>(k) *
                         static_cast<double>(max_abs_a) *
                         static_cast<double>(max_abs_b) *
                         scalar_factor;

    const double out_max = static_cast<double>((std::numeric_limits<int32_t>::max)());
    if (!(bound > out_max)) {
        return;
    }

    std::string message;
    message.reserve(512);
    message += "pycauset matmul: int32 @ int32 may overflow the int32 output (advisory estimate from sampled entries). ";
    message += "m=" + std::to_string(m);
    message += ", k=" + std::to_string(k);
    message += ", n=" + std::to_string(n);
    message += ", sampled max|A|=" + std::to_string(max_abs_a);
    message += ", sampled max|B|=" + std::to_string(max_abs_b);
    message += ", estimated bound ~" + std::to_string(bound);
    message += ". The result is computed exactly; PyCauset raises OverflowError if a value does not fit.";
    bindings_warn::warn_with_category(message,
        "PyCausetOverflowRiskWarning",
        /*stacklevel=*/3);
}


using namespace pycauset_bindings::indexing;  // matrix_indexing.hpp

// Bitwise ops (math/Bitwise.cpp): a float/complex operand is a TypeError (pycauset::dtype_error,
// translated globally), as in NumPy.
template <typename F>
inline std::shared_ptr<MatrixBase> bitwise_call(F&& f) {
    return translate_invalid_argument([&] { return std::shared_ptr<MatrixBase>(f().release()); });
}

inline std::shared_ptr<MatrixBase> matrix_inverse(const MatrixBase& mat) {
    return std::shared_ptr<MatrixBase>(pycauset::inverse(mat).release());  // one entry (LinearAlgebra.cpp)
}

// Construct a structure of a dtype given by name, through the one factory.
template <typename StructureBase>
std::shared_ptr<StructureBase> make_structure(uint64_t rows, uint64_t cols, pycauset::Structure s, const py::object& dtype) {
    auto obj = pycauset::ObjectFactory::create_matrix(rows, cols, dtype_arg(dtype), s, "");
    auto* typed = dynamic_cast<StructureBase*>(obj.get());
    if (typed == nullptr) throw std::logic_error("factory returned the wrong structure");
    obj.release();
    return std::shared_ptr<StructureBase>(typed);
}

inline py::object symmetric_from_triangular(const MatrixBase& source, pycauset::Structure kind) {
    if (source.get_structure() != pycauset::Structure::TRIANGULAR) {
        throw py::type_error("from_triangular expects a TriangularMatrix");
    }
    if (source.is_transposed()) {
        throw py::value_error("from_triangular expects an upper triangle; this one is lower (use its .T)");
    }
    return pycauset::visit_matrix(source, [&](const auto& tri) -> py::object {
        using M = std::decay_t<decltype(tri)>;
        using E = element_of_t<M>;
        if constexpr (std::is_same_v<M, TriangularMatrix<E>> && !std::is_same_v<E, bool>) {
            return py::cast(std::shared_ptr<MatrixBase>(SymmetricMatrix<E>::from_triangular(tri, kind, "").release()));
        } else {
            throw py::type_error("from_triangular is not supported for " + describe(source));
        }
    });
}

// Re-open stored data: the constructors take the *storage* shape and the view flags. Which
// (structure, dtype) pairs exist is the catalogue's decision (DTypeDispatch.hpp).
inline py::object open_storage(pycauset::Structure s, DataType dt, uint64_t rows, uint64_t cols,
                               const std::string& path, size_t offset, uint64_t seed,
                               std::complex<double> scalar, bool is_transposed, bool has_diagonal) {
    using S = pycauset::Structure;
    if (s == S::VECTOR) {
        return pycauset::visit_dtype(dt, [&](auto tag) -> py::object {
            using V = typename pycauset::DenseVectorOf<typename decltype(tag)::type>::type;
            return py::cast(std::shared_ptr<VectorBase>(std::make_shared<V>(rows, path, offset, seed, scalar, is_transposed)));
        });
    }
    if (s == S::UNIT_VECTOR) {
        return py::cast(std::shared_ptr<VectorBase>(std::make_shared<UnitVector>(rows, seed, path, offset, seed, scalar, is_transposed, dt)));
    }
    pycauset::MatrixOptions opts;
    opts.has_diagonal = has_diagonal;
    auto m = pycauset::open_matrix(s, dt, rows, cols, path, offset, seed, scalar, is_transposed, opts);
    return py::cast(std::shared_ptr<MatrixBase>(m.release()));
}


// ---------------------------------------------------------------------------
// NumPy export (to_numpy, np.asarray, __array__), one path for every class, dtype and view:
// the logical values as plain dense storage of the matrix's dtype (the shared typed reader:
// to_dense / write_out; no copy for plain dense storage), then one copy into the array.
// bool: packed bits unpacked to bytes; float16: the uint16 bits viewed as float16;
// complex_float16: complex64 (NumPy has no complex32). Replaces three exporters (dense-only
// typed, generic per-element, and __array__ with a float64 default for unknown classes).
// ---------------------------------------------------------------------------
py::object export_numpy(const MatrixBase& mat) {
    const uint64_t rows = mat.rows(), cols = mat.cols();
    const DataType dt = mat.get_data_type();
    const bool plain = mat.get_structure() == pycauset::Structure::DENSE && pycauset::is_plain(mat) &&
                       rows == mat.base_rows() && cols == mat.base_cols() &&
                       dynamic_cast<const pycauset::BitClosureMatrix*>(&mat) == nullptr;
    std::unique_ptr<MatrixBase> h;
    if (!plain) {
        py::gil_scoped_release release;
        h = dt == DataType::BIT ? pycauset::write_out(mat) : pycauset::to_dense(mat, dt);
    }
    const MatrixBase& src = h ? *h : mat;
    const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)};
    if (dt == DataType::BIT) {
        py::array_t<bool> out(shape);
        bool* dst = out.mutable_data();
        const auto& b = static_cast<const DenseMatrix<bool>&>(src);
        const uint64_t* words = b.data();
        const uint64_t stride = b.stride_bytes() / 8;
        py::gil_scoped_release release;
        pycauset::ParallelFor(0, rows, [&](size_t i) {
            const uint64_t* w = words + i * stride;
            for (uint64_t j = 0; j < cols; ++j) dst[i * cols + j] = (w[j / 64] >> (j % 64)) & 1u;
        });
        return out;
    }
    if (dt == DataType::COMPLEX_FLOAT16) {
        py::array_t<std::complex<float>> out(shape);
        std::complex<float>* dst = out.mutable_data();
        const auto& c = static_cast<const pycauset::ComplexFloat16Matrix&>(src);
        const pycauset::float16_t* re = c.real_data();
        const pycauset::float16_t* im = c.imag_data();
        for (uint64_t k = 0; k < rows * cols; ++k) dst[k] = {static_cast<float>(re[k]), static_cast<float>(im[k])};
        return out;
    }
    static const std::map<DataType, const char*> names = {
        {DataType::INT8, "int8"}, {DataType::INT16, "int16"}, {DataType::INT32, "int32"}, {DataType::INT64, "int64"},
        {DataType::UINT8, "uint8"}, {DataType::UINT16, "uint16"}, {DataType::UINT32, "uint32"},
        {DataType::UINT64, "uint64"}, {DataType::FLOAT16, "float16"}, {DataType::FLOAT32, "float32"},
        {DataType::FLOAT64, "float64"}, {DataType::COMPLEX_FLOAT32, "complex64"},
        {DataType::COMPLEX_FLOAT64, "complex128"}};
    const auto it = names.find(dt);
    if (it == names.end()) throw py::type_error(std::string("no NumPy dtype for ") + pycauset::dtype_name(dt));
    py::array out(py::dtype(it->second), shape);
    const void* data = src.shared_mapper()->get_data();
    std::memcpy(out.mutable_data(), data, static_cast<size_t>(rows * cols) * static_cast<size_t>(out.itemsize()));
    return out;
}

// Matrix arithmetic, the single operator path (PLAN 4.3): `a op other` (`other op a` when
// reflected) for another matrix, any Python/NumPy scalar, or a NumPy array. Scalars go to
// scalar_op (NumPy result dtypes, exact integers, O(1) float views); matrices to the
// element-wise ops. Anything else returns NotImplemented (e.g. BlockMatrix handles it).
// One matmul binding path (review 4.3): `@`, `.matmul`, `pycauset._native.matmul` and NumPy
// operands all come here. The warnings are advisory and never change the result.
inline std::shared_ptr<MatrixBase> matmul_mm(const MatrixBase& a, const MatrixBase& b) {
    return translate_invalid_argument([&]() {
        maybe_warn_float_underpromotion("matmul", promotion::BinaryOp::Matmul, a, b);
        maybe_warn_bit_promotes_to_int32("matmul", promotion::BinaryOp::Matmul, a.get_data_type(), b.get_data_type());
        maybe_warn_integer_matmul_overflow_risk_preflight(a, b);
        return std::shared_ptr<MatrixBase>(pycauset::dispatch_matmul(a, b, "").release());
    });
}

inline std::shared_ptr<VectorBase> matvec(const MatrixBase& a, const VectorBase& v) {
    return translate_invalid_argument([&]() {
        maybe_warn_bit_promotes_to_int32("matmul", promotion::BinaryOp::MatrixVectorMultiply, a.get_data_type(),
                                         v.get_data_type());
        return std::shared_ptr<VectorBase>(pycauset::matrix_vector_multiply(a, v, "").release());
    });
}

inline py::object matrix_arith(const MatrixBase& a, const py::object& other, Arith op, bool reflected) {
    std::shared_ptr<MatrixBase> b;
    if (py::isinstance<MatrixBase>(other)) {
        b = other.cast<std::shared_ptr<MatrixBase>>();
    } else if (pycauset::ScalarArg s; to_scalar_arg(other, s)) {
        auto out = pycauset::scalar_op(a, scalar_kind(op, reflected), s);
        // A NumPy scalar wider than the result (float32 M + np.float64(1) is float32): the same
        // notice as for two matrices (DType System §9).
        if (s.strong && float_precision_bits(s.strong_dtype) > float_precision_bits(out->get_data_type()) &&
            float_precision_bits(out->get_data_type()) > 0) {
            bindings_warn::warn_with_category(
                std::string("pycauset: a ") + pycauset::dtype_name(s.strong_dtype) + " scalar with a " +
                    pycauset::dtype_name(a.get_data_type()) + " matrix gives " + pycauset::dtype_name(out->get_data_type()) +
                    " (computed and stored in it; NumPy would widen). Cast the matrix first to compute in the wider dtype.",
                "PyCausetDTypeWarning", /*stacklevel=*/2);
        }
        return py::cast(std::shared_ptr<MatrixBase>(out.release()));
    } else if (py::isinstance<py::array>(other)) {
        b = elementwise_operand_matrix_from_numpy(other.cast<py::array>());
    } else if (py::isinstance<VectorBase>(other)) {
        // A 1-D operand broadcasts as a row, as in NumPy (F-102): the vector's 1 x n view.
        b = std::shared_ptr<MatrixBase>(pycauset::as_row_matrix(other.cast<const VectorBase&>()).release());
    } else {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    const MatrixBase& x = reflected ? *b : a;
    const MatrixBase& y = reflected ? a : *b;
    std::unique_ptr<MatrixBase> out;
    switch (op) {
        case Arith::Add:
            maybe_warn_float_underpromotion("add", promotion::BinaryOp::Add, x, y);
            maybe_warn_bit_promotes_to_int32("add", promotion::BinaryOp::Add, x.get_data_type(), y.get_data_type());
            out = pycauset::add(x, y, "");
            break;
        case Arith::Sub:
            maybe_warn_float_underpromotion("subtract", promotion::BinaryOp::Subtract, x, y);
            maybe_warn_bit_promotes_to_int32("subtract", promotion::BinaryOp::Subtract, x.get_data_type(), y.get_data_type());
            out = pycauset::subtract(x, y, "");
            break;
        case Arith::Mul:
            maybe_warn_float_underpromotion("elementwise_multiply", promotion::BinaryOp::ElementwiseMultiply, x, y);
            out = pycauset::elementwise_multiply(x, y, "");
            break;
        case Arith::Div:
            maybe_warn_float_underpromotion("elementwise_divide", promotion::BinaryOp::Divide, x, y);
            out = pycauset::elementwise_divide(x, y, "");
            break;
    }
    return py::cast(std::shared_ptr<MatrixBase>(out.release()));
}

} // namespace

namespace pycauset_bindings {
py::object export_matrix_numpy(const pycauset::MatrixBase& m) { return export_numpy(m); }
}  // namespace pycauset_bindings

void bind_matrix_classes(py::module_& m) {
    auto matrix_base_class = py::class_<MatrixBase, std::shared_ptr<MatrixBase>>(m, "MatrixBase", py::dynamic_attr());
    matrix_base_class
        .def_property_readonly("__array_priority__", [](const MatrixBase&) { 
            return 1000.0; 
        })
        .def_property_readonly("backing_file", &MatrixBase::get_backing_file)
        .def("get_backing_file", &MatrixBase::get_backing_file)
        .def_property_readonly("is_temporary", &MatrixBase::is_temporary)
        .def("set_temporary", &MatrixBase::set_temporary)
        .def("close", &MatrixBase::close)
        // Read-ahead / drop hints for the whole storage. The accelerator is looked up per call:
        // a handle kept in Python dangled after the next storage swap (review 3.1 C-16).
        .def("_io_prefetch", [](const MatrixBase& m) {
            if (auto* acc = m.get_accelerator()) acc->prefetch(0, SIZE_MAX);
        })
        .def("_io_discard", [](const MatrixBase& m) {
            if (auto* acc = m.get_accelerator()) acc->discard(0, SIZE_MAX);
        })
        .def("hint", &MatrixBase::hint, py::arg("hint"))
        .def_property("seed", &MatrixBase::get_seed, &MatrixBase::set_seed)
        .def_property("scalar", &MatrixBase::get_scalar, &MatrixBase::set_scalar)
        .def("get_scalar", &MatrixBase::get_scalar)
        .def("set_scalar", &MatrixBase::set_scalar)
        .def("is_transposed", &MatrixBase::is_transposed)
        .def("set_transposed", &MatrixBase::set_transposed)
        .def("is_conjugated", &MatrixBase::is_conjugated)
        .def("set_conjugated", &MatrixBase::set_conjugated)
        .def_property_readonly(
            "T",
            [](const MatrixBase& mat) {
                auto out = mat.transpose("");
                return std::shared_ptr<MatrixBase>(out.release());
            })
        .def_property_readonly(
            "H",
            [](const MatrixBase& mat) {
                // A^H = conj(A^T): toggle the transposed result's flag (a Hermitian transpose
                // already toggles it; using `mat`'s flag dropped that, so A.H was wrong).
                auto out = mat.transpose("");
                out->set_conjugated(!out->is_conjugated());
                return std::shared_ptr<MatrixBase>(out.release());
            })
        .def(
            "transpose",
            [](const MatrixBase& mat) {
                auto out = mat.transpose("");
                return std::shared_ptr<MatrixBase>(out.release());
            })
        .def(
            "conj",
            [](const MatrixBase& mat) {
                auto out = mat.clone();
                auto* m = dynamic_cast<MatrixBase*>(out.get());
                if (!m) {
                    throw std::runtime_error("Internal error: clone() did not return a MatrixBase");
                }
                m->set_conjugated(!mat.is_conjugated());
                return std::shared_ptr<MatrixBase>(static_cast<MatrixBase*>(out.release()));
            })
        .def_property_readonly("shape", [](const MatrixBase& mat) {
            return py::make_tuple(mat.rows(), mat.cols());
        })
        .def_property_readonly("size", &MatrixBase::size, "Number of elements (rows * cols), as NumPy's ndarray.size.")
        .def("rows", &MatrixBase::rows)
        .def("cols", &MatrixBase::cols)
        .def("get_element_as_double", &MatrixBase::get_element_as_double)
        .def("get_element_as_complex", &MatrixBase::get_element_as_complex)
        .def("_to_numpy_fast", [](const MatrixBase& mat, bool allow_huge) -> py::object {
            if (!allow_huge) {
                pycauset::bindings::ensure_numpy_export_allowed(mat);
            }
            return export_numpy(mat);
        }, py::arg("allow_huge") = false)
        .def("trace", [](const MatrixBase& mat) { return pycauset_bindings::reduction_to_py(pycauset::trace(mat)); })
        .def("determinant", [](const MatrixBase& mat) -> py::object {
            // NumPy kinds: complex for complex matrices, float otherwise (F-059).
            const std::complex<double> d = pycauset::determinant(mat);
            const DataType dt = mat.get_data_type();
            const bool cplx = is_complex_dtype(dt) || mat.get_scalar().imag() != 0.0;
            return cplx ? py::cast(d) : py::cast(d.real());
        })
        .def("slogdet", [](const MatrixBase& mat) -> py::tuple {
            const auto [sign, logabs] = pycauset::slogdet(mat);
            const DataType dt = mat.get_data_type();
            const bool cplx = is_complex_dtype(dt) || mat.get_scalar().imag() != 0.0;
            return py::make_tuple(cplx ? py::cast(sign) : py::cast(sign.real()), logabs);
        })
        .def(
            "__matmul__",
            [](const MatrixBase& a, const std::shared_ptr<MatrixBase>& b) { return matmul_mm(a, *b); },
            py::is_operator())
        .def(
            "__matmul__",
            [](const MatrixBase& a, const std::shared_ptr<VectorBase>& v) { return matvec(a, *v); },
            py::is_operator())
        .def(
            "__matmul__",
            [](const MatrixBase& a, const py::array& b) -> py::object {
                if (b.ndim() == 1) return py::cast(matvec(a, *vector_from_numpy(b)));
                if (b.ndim() == 2) return py::cast(matmul_mm(a, *matrix_from_numpy(b)));
                throw py::value_error("Unsupported array rank for matmul");
            },
            py::is_operator())
        .def(
            "__rmatmul__",
            // ndarray @ matrix: NumPy defers (`__array_ufunc__ = None`). A 1-D array is a row
            // vector (NumPy's vector @ matrix), returned as a vector.
            [](const MatrixBase& a, const py::array& b) -> py::object {
                if (b.ndim() == 1) {
                    return py::cast(translate_invalid_argument([&]() {
                        auto out = pycauset::vector_matrix_multiply(*vector_from_numpy(b), a, "");
                        out->set_transposed(false);  // a 1-D result, as NumPy's (not a 1 x n row)
                        return std::shared_ptr<VectorBase>(out.release());
                    }));
                }
                if (b.ndim() == 2) return py::cast(matmul_mm(*matrix_from_numpy(b), a));
                throw py::value_error("Unsupported array rank for matmul");
            },
            py::is_operator())
        .def(
            "__matmul__",
            [](const MatrixBase&, const py::object&) -> py::object {
                // Allow other operand types (e.g., internal BlockMatrix) to
                // handle the operation via their __rmatmul__.
                return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            },
            py::is_operator())
        .def("__add__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Add, false); }, py::is_operator())
        .def("__radd__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Add, true); }, py::is_operator())
        .def("__sub__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Sub, false); }, py::is_operator())
        .def("__rsub__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Sub, true); }, py::is_operator())
        .def("__mul__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Mul, false); }, py::is_operator())
        .def("__rmul__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Mul, true); }, py::is_operator())
        .def("__truediv__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Div, false); }, py::is_operator())
        .def("__rtruediv__", [](const MatrixBase& a, const py::object& b) { return matrix_arith(a, b, Arith::Div, true); }, py::is_operator())
        .def(
            "__neg__",
            [](const MatrixBase& a) {
                // F-002: unary minus (wraps for unsigned ints like NumPy; bit/bool raises).
                return std::shared_ptr<MatrixBase>(pycauset::negate(a).release());
            },
            py::is_operator())
        .def(
            "__pos__",
            [](const MatrixBase& a) {
                return std::shared_ptr<MatrixBase>(a.multiply_scalar(static_cast<int64_t>(1)).release());
            },
            py::is_operator());


    // Encourage NumPy to prefer MatrixBase reverse ops over coercion.
    py::object mb_cls = m.attr("MatrixBase");
    mb_cls.attr("__array_priority__") = py::float_(1000.0);

    // Native-side export ceiling so NumPy coercion paths (buffer protocol / native __array__)
    // honor the same policy as pycauset.to_numpy(...).
    m.def(
        "_set_numpy_export_max_bytes",
        [](py::object limit) {
            if (limit.is_none()) {
                pycauset::bindings::g_numpy_export_max_bytes.store(-1);
                return;
            }
            const int64_t v = limit.cast<int64_t>();
            pycauset::bindings::g_numpy_export_max_bytes.store(v);
        },
        py::arg("limit"));

    // ------------------------------------------------------------------------------------
    // One Python class per structure; the dtype is an attribute (RESTRUCTURE 3.1 item 4).
    // Every method is generic over dtype through pycauset::visit_matrix (DTypeDispatch.hpp);
    // the polymorphic type hook in bindings_common.hpp maps each object to its class.
    // ------------------------------------------------------------------------------------
    matrix_base_class
        .def_property_readonly(
            "dtype",
            [](const MatrixBase& mat) { return std::string(pycauset::dtype_name(mat.get_data_type())); },
            "Element type, e.g. 'float64', 'int32', 'bool' (bit), 'complex_float64'.")
        .def("get", [](const MatrixBase& mat, uint64_t i, uint64_t j) { return element_get(mat, i, j); },
             py::arg("i"), py::arg("j"))
        .def("set", [](MatrixBase& mat, uint64_t i, uint64_t j, const py::object& value) { element_set(mat, i, j, value); },
             py::arg("i"), py::arg("j"), py::arg("value"))
        .def("__getitem__", [](const MatrixBase& mat, const py::object& key) -> py::object {
            return matrixbase_getitem_dispatch(mat, key);
        })
        .def("__setitem__", [](MatrixBase& mat, const py::object& key, const py::object& value) {
            matrixbase_setitem_dispatch(mat, key, value);
        })
        .def("fill", [](MatrixBase& mat, const py::object& value) {
            pycauset::visit_matrix(mat, [&](auto& typed) {
                using M = std::remove_reference_t<decltype(typed)>;
                if constexpr (requires { typed.fill(element_from_py<M>(value)); }) {
                    typed.fill(element_from_py<M>(value));
                } else {
                    throw py::type_error(std::string("fill() is not supported for ") + describe(mat));
                }
            });
        }, py::arg("value"))
        .def("__invert__", [](const MatrixBase& a) { return bitwise_call([&] { return pycauset::bitwise_not(a); }); },
             py::is_operator())
        .def("__and__", [](const MatrixBase& a, const MatrixBase& b) {
                 return bitwise_call([&] { return pycauset::bitwise(a, b, pycauset::BitwiseOp::And); });
             }, py::is_operator())
        .def("__or__", [](const MatrixBase& a, const MatrixBase& b) {
                 return bitwise_call([&] { return pycauset::bitwise(a, b, pycauset::BitwiseOp::Or); });
             }, py::is_operator())
        .def("__xor__", [](const MatrixBase& a, const MatrixBase& b) {
                 return bitwise_call([&] { return pycauset::bitwise(a, b, pycauset::BitwiseOp::Xor); });
             }, py::is_operator())
        .def_property_readonly("is_lazy", [](const MatrixBase& mat) {
            if (auto* c = dynamic_cast<const pycauset::BitClosureMatrix*>(&mat)) {
                if (!c->is_materialized()) return true;
            }
            return mat.get_scalar() != std::complex<double>(1.0, 0.0) || mat.is_transposed() || mat.is_conjugated();
        }, "True when the elements are computed on read (a view or closure) rather than stored.")
        .def("matmul", [](const MatrixBase& a, const MatrixBase& b) { return matmul_mm(a, b); }, py::arg("other"))
        .def("eig", [](const MatrixBase& mat) {
            auto r = pycauset::eig(mat);
            return py::make_tuple(std::shared_ptr<VectorBase>(r.first.release()), std::shared_ptr<MatrixBase>(r.second.release()));
        })
        .def("eigvals", [](const MatrixBase& mat) { return std::shared_ptr<VectorBase>(pycauset::eigvals(mat).release()); })
        .def("eigh", [](const MatrixBase& mat) {
            auto r = pycauset::eigh(mat);
            return py::make_tuple(std::shared_ptr<VectorBase>(r.first.release()), std::shared_ptr<MatrixBase>(r.second.release()));
        })
        .def("eigvalsh", [](const MatrixBase& mat) { return std::shared_ptr<VectorBase>(pycauset::eigvalsh(mat).release()); });

    // --- DenseMatrix ---------------------------------------------------------------------
    py::class_<pycauset::DenseMatrixBase, MatrixBase, std::shared_ptr<pycauset::DenseMatrixBase>>(
        m, "DenseMatrix", "Dense matrix of any dtype. Prefer pycauset.matrix / zeros / empty to construct.")
        .def(py::init([](uint64_t rows, std::optional<uint64_t> cols, const py::object& dtype) {
                 return make_structure<pycauset::DenseMatrixBase>(
                     rows, cols.value_or(rows), pycauset::Structure::DENSE, dtype);
             }),
             py::arg("rows"), py::arg("cols") = std::nullopt, py::arg("dtype") = "float64")
        .def("set_identity", [](pycauset::DenseMatrixBase& mat) {
            pycauset::visit_matrix(mat, [&](auto& typed) {
                if constexpr (requires { typed.set_identity(); }) {
                    typed.set_identity();
                } else {
                    throw py::type_error(std::string("set_identity() is not supported for ") + describe(mat));
                }
            });
        })
        .def_static(
            "random",
            [](uint64_t rows, std::optional<uint64_t> cols, double p, std::optional<uint64_t> seed) {
                auto out = DenseBitMatrix::random(rows, cols.value_or(rows), p, "", seed);
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::arg("rows"), py::arg("cols") = std::nullopt, py::arg("p") = 0.5, py::arg("seed") = std::nullopt,
            "Random bit (dtype 'bool') matrix; each entry is 1 with probability p.");

    // --- TriangularMatrix (strictly upper; a causal matrix is TriangularMatrix of dtype 'bool') ---
    py::class_<TriangularMatrixBase, MatrixBase, std::shared_ptr<TriangularMatrixBase>>(
        m, "TriangularMatrix", "Upper-triangular matrix (strict unless has_diagonal). Causal matrices have dtype 'bool'.")
        .def(py::init([](uint64_t n, const py::object& dtype, bool has_diagonal) {
                 pycauset::MatrixOptions opts;
                 opts.has_diagonal = has_diagonal;
                 auto m = pycauset::make_matrix(pycauset::Structure::TRIANGULAR, dtype_arg(dtype), n, n, "", opts);
                 return std::shared_ptr<TriangularMatrixBase>(static_cast<TriangularMatrixBase*>(m.release()));
             }),
             py::arg("n"), py::arg("dtype") = "float64", py::arg("has_diagonal") = false)
        .def_property_readonly("has_diagonal", &TriangularMatrixBase::has_diagonal)
        .def_static(
            "random",
            [](uint64_t n, double p, std::optional<uint64_t> seed) {
                auto out = TriangularBitMatrix::random(n, p, "", seed);
                return std::shared_ptr<MatrixBase>(out.release());
            },
            py::arg("n"), py::arg("p") = 0.5, py::arg("seed") = std::nullopt,
            "Random strictly upper-triangular bit (dtype 'bool') matrix; each entry is 1 with probability p.");

    // --- DiagonalMatrix / IdentityMatrix ---------------------------------------------------
    py::class_<pycauset::DiagonalMatrixBase, MatrixBase, std::shared_ptr<pycauset::DiagonalMatrixBase>>(
        m, "DiagonalMatrix", "Diagonal matrix (stores n elements).")
        .def(py::init([](uint64_t n, const py::object& dtype) {
                 return make_structure<pycauset::DiagonalMatrixBase>(n, n, pycauset::Structure::DIAGONAL, dtype);
             }),
             py::arg("n"), py::arg("dtype") = "float64")
        .def("get_diagonal", [](const pycauset::DiagonalMatrixBase& mat, uint64_t i) { return element_get(mat, i, i); },
             py::arg("i"))
        .def("set_diagonal", [](pycauset::DiagonalMatrixBase& mat, uint64_t i, const py::object& v) { element_set(mat, i, i, v); },
             py::arg("i"), py::arg("value"));

    py::class_<pycauset::IdentityMatrixBase, MatrixBase, std::shared_ptr<pycauset::IdentityMatrixBase>>(
        m, "IdentityMatrix", "Identity matrix (no element storage; read-only).")
        .def(py::init([](uint64_t rows, std::optional<uint64_t> cols, const py::object& dtype) {
                 return make_structure<pycauset::IdentityMatrixBase>(
                     rows, cols.value_or(rows), pycauset::Structure::IDENTITY, dtype);
             }),
             py::arg("rows"), py::arg("cols") = std::nullopt, py::arg("dtype") = "float64");

    // --- The packed symmetric family (one C++ class with a kind, four Python classes) -------
    // Every SymmetricMatrix<T> derives all four C++ markers; Python keeps the classes unrelated
    // (an antisymmetric matrix is not symmetric; a complex symmetric matrix is not Hermitian),
    // hence MatrixBase as the base. The structure picks the class (bindings_common.hpp).
    auto bind_symmetric_kind = [&](auto marker, const char* name, pycauset::Structure kind, const char* doc,
                                   const char* from_doc) {
        using B = typename decltype(marker)::type;
        py::class_<B, MatrixBase, std::shared_ptr<B>>(m, name, doc)
            .def(py::init([kind](uint64_t n, const py::object& dtype) {
                     return make_structure<B>(n, n, kind, dtype);
                 }),
                 py::arg("n"), py::arg("dtype") = "float64")
            .def_static("from_triangular", [kind](const MatrixBase& source) { return symmetric_from_triangular(source, kind); },
                        py::arg("source"), from_doc);
    };
    bind_symmetric_kind(pycauset::type_tag<pycauset::SymmetricMatrixBase>{}, "SymmetricMatrix",
                        pycauset::Structure::SYMMETRIC, "Symmetric matrix, A == A.T (packed upper triangle).",
                        "Symmetric matrix whose upper triangle is `source`'s.");
    bind_symmetric_kind(pycauset::type_tag<pycauset::AntiSymmetricMatrixBase>{}, "AntiSymmetricMatrix",
                        pycauset::Structure::ANTISYMMETRIC,
                        "Antisymmetric matrix, A == -A.T (packed strict upper triangle; signed dtypes).",
                        "Antisymmetric matrix whose strict upper triangle is `source`'s.");
    bind_symmetric_kind(pycauset::type_tag<pycauset::HermitianMatrixBase>{}, "HermitianMatrix",
                        pycauset::Structure::HERMITIAN,
                        "Hermitian matrix, A == A.conj().T (packed upper triangle, real diagonal; complex dtypes). "
                        "Not the same as a complex SymmetricMatrix (A == A.T).",
                        "Hermitian matrix whose upper triangle is `source`'s (its diagonal must be real).");
    bind_symmetric_kind(pycauset::type_tag<pycauset::SkewHermitianMatrixBase>{}, "SkewHermitianMatrix",
                        pycauset::Structure::SKEW_HERMITIAN,
                        "Skew-Hermitian matrix, A == -A.conj().T (packed upper triangle, imaginary diagonal; "
                        "complex dtypes).",
                        "Skew-Hermitian matrix whose upper triangle is `source`'s (its diagonal must be imaginary).");

    // --- Re-open stored data (persistence.load) ---------------------------------------------
    m.def(
        "_open_storage",
        [](const std::string& structure, const std::string& dtype, uint64_t rows, uint64_t cols,
           const std::string& path, size_t offset, uint64_t seed, std::complex<double> scalar,
           bool is_transposed, bool has_diagonal) -> py::object {
            return open_storage(pycauset::parse_structure(structure), pycauset::parse_dtype_name(dtype), rows, cols,
                                path, offset, seed, scalar, is_transposed, has_diagonal);
        },
        py::arg("structure"), py::arg("dtype"), py::arg("rows"), py::arg("cols"), py::arg("path"),
        py::arg("offset"), py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"),
        py::arg("has_diagonal") = false,
        "Internal: map an existing data file as a matrix or vector (storage shape; the view flags are applied).");
    m.def("_materialize_for_save", [](const std::shared_ptr<MatrixBase>& mat) -> std::shared_ptr<MatrixBase> {
        // Closure views read another object's storage layout; files store the packed result.
        if (auto* c = dynamic_cast<const pycauset::BitClosureMatrix*>(mat.get())) {
            return std::shared_ptr<MatrixBase>(c->materialized_copy().release());
        }
        // A window's file must hold the window, not the whole storage it looks into (the view
        // record has no offsets): saving A[1:4, 1:4] stored all of A (F-072).
        if (mat->is_subview()) return std::shared_ptr<MatrixBase>(pycauset::write_out(*mat).release());
        return mat;
    }, py::arg("matrix"));
    m.def("materialize", [](const std::shared_ptr<MatrixBase>& mat) -> std::shared_ptr<MatrixBase> {
        if (auto* c = dynamic_cast<const pycauset::BitClosureMatrix*>(mat.get())) {
            return std::shared_ptr<MatrixBase>(c->materialized_copy().release());
        }
        if (mat->get_scalar() != std::complex<double>(1.0, 0.0) || mat->is_transposed() || mat->is_conjugated()) {
            return std::shared_ptr<MatrixBase>(pycauset::materialize(*mat).release());
        }
        return mat;
    }, py::arg("matrix"), "The matrix with its elements stored (views and closures evaluated).");
    m.def("_debug_fill", [](MatrixBase& mat, py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr) {
        // Test-harness helper (tests/atlas/timing.py): write the stored entries of `mat` from a
        // logical array in one typed C++ loop, instead of n^2 Python `m[i, j] = v` calls.
        const Structure st = mat.get_structure();
        if (st == Structure::IDENTITY) return;
        auto a = arr.unchecked<2>();
        const uint64_t rows = mat.rows(), cols = mat.cols();
        if (static_cast<uint64_t>(a.shape(0)) != rows || static_cast<uint64_t>(a.shape(1)) != cols) {
            throw py::value_error("_debug_fill: shape mismatch");
        }
        const bool diag = st == Structure::TRIANGULAR && static_cast<const TriangularMatrixBase&>(mat).has_diagonal();
        pycauset::visit_matrix(mat, [&](auto& typed) {
            using T = std::decay_t<decltype(typed.get(0, 0))>;
            if constexpr (requires { typed.set(0, 0, T{}); }) {
                for (uint64_t i = 0; i < rows; ++i) {
                    for (uint64_t j = 0; j < cols; ++j) {
                        const bool stored = st == Structure::DENSE ? true
                                          : st == Structure::TRIANGULAR ? (diag ? j >= i : j > i)
                                          : (st == Structure::SYMMETRIC || st == Structure::HERMITIAN ||
                                             st == Structure::SKEW_HERMITIAN) ? j >= i
                                          : st == Structure::ANTISYMMETRIC ? j > i
                                          : st == Structure::DIAGONAL ? i == j : false;
                        const std::complex<double> z = a(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j));
                        if (!stored || z == std::complex<double>(0.0, 0.0)) continue;
                        if constexpr (std::is_same_v<T, bool>) typed.set(i, j, z.real() != 0.0);
                        else if constexpr (pycauset::is_complex_element_v<T> || std::is_same_v<T, std::complex<double>>)
                            typed.set(i, j, T(z));
                        else typed.set(i, j, static_cast<T>(z.real()));
                    }
                }
            } else {
                throw py::type_error("_debug_fill: matrix class has no typed set");
            }
        });
    }, py::arg("matrix"), py::arg("values"));
    m.def("_debug_is_closure", [](const MatrixBase& mat) {
        return dynamic_cast<const pycauset::BitClosureMatrix*>(&mat) != nullptr;
    }, py::arg("matrix"), "True for a lazy closure result (its work happens when it is evaluated).");
    m.def("_pack_as", [](const MatrixBase& mat, const std::string& structure, bool has_diagonal) {
        return std::shared_ptr<MatrixBase>(
            pycauset::pack_as(mat, pycauset::parse_structure(structure), has_diagonal).release());
    }, py::arg("matrix"), py::arg("structure"), py::arg("has_diagonal") = false,
       "Internal: the stored region of `matrix` as a new matrix of `structure` (same dtype).");
    m.def("_structure_of", [](const py::object& obj) -> std::string {
        if (py::isinstance<MatrixBase>(obj)) return pycauset::structure_name(obj.cast<const MatrixBase&>().get_structure());
        if (py::isinstance<VectorBase>(obj)) return pycauset::structure_name(obj.cast<const VectorBase&>().get_structure());
        throw py::type_error("not a pycauset matrix or vector");
    }, py::arg("obj"), "Internal: the storage structure name ('dense', 'triangular', ...).");

    m.def("asarray", [](const py::array& array) -> py::object {
        auto buf = array.request();
        if (buf.ndim == 1) {
            if (is_numpy_float16(array)) {
                auto b = array.request();
                require_1d(b);
                uint64_t n = static_cast<uint64_t>(b.shape[0]);
                auto result = std::make_shared<DenseVector<pycauset::float16_t>>(n);
                const uint16_t* src_ptr = static_cast<const uint16_t*>(b.ptr);
                pycauset::float16_t* dst_ptr = result->data();
                if (b.strides[0] == static_cast<py::ssize_t>(sizeof(uint16_t))) {
                    std::memcpy(static_cast<void*>(dst_ptr), src_ptr, n * sizeof(uint16_t));
                } else {
                    const ptrdiff_t stride0 = static_cast<ptrdiff_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                    for (uint64_t i = 0; i < n; ++i) {
                        const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0;
                        dst_ptr[i] = pycauset::float16_t(static_cast<uint16_t>(src_ptr[idx]));
                    }
                }
                return py::cast(result);
            }
            if (py::isinstance<py::array_t<double>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<double>(array.cast<py::array_t<double>>()));
            }
            if (py::isinstance<py::array_t<float>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<float>(array.cast<py::array_t<float>>()));  // dtype kept (F-056)
            }
            if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<std::complex<float>>(array.cast<py::array_t<std::complex<float>>>()));
            }
            if (py::isinstance<py::array_t<std::complex<double>>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<std::complex<double>>(array.cast<py::array_t<std::complex<double>>>()));
            }
            if (py::isinstance<py::array_t<int32_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<int32_t>(array.cast<py::array_t<int32_t>>()));
            }
            if (py::isinstance<py::array_t<int64_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<int64_t>(array.cast<py::array_t<int64_t>>()));
            }
            if (py::isinstance<py::array_t<int8_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<int8_t>(array.cast<py::array_t<int8_t>>()));
            }
            if (py::isinstance<py::array_t<int16_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<int16_t>(array.cast<py::array_t<int16_t>>()));
            }
            if (py::isinstance<py::array_t<uint8_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<uint8_t>(array.cast<py::array_t<uint8_t>>()));
            }
            if (py::isinstance<py::array_t<uint16_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<uint16_t>(array.cast<py::array_t<uint16_t>>()));
            }
            if (py::isinstance<py::array_t<uint32_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<uint32_t>(array.cast<py::array_t<uint32_t>>()));
            }
            if (py::isinstance<py::array_t<uint64_t>>(array)) {
                return py::cast(dense_vector_from_numpy_1d<uint64_t>(array.cast<py::array_t<uint64_t>>()));
            }
            if (py::isinstance<py::array_t<bool>>(array)) {
                auto tmp = array.cast<py::array_t<bool>>();
                auto b = tmp.request();
                require_1d(b);
                uint64_t n = static_cast<uint64_t>(b.shape[0]);
                auto result = std::make_shared<DenseVector<bool>>(n);
                const bool* src_ptr = static_cast<const bool*>(b.ptr);
                const ptrdiff_t stride0 = static_cast<ptrdiff_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
                for (uint64_t i = 0; i < n; ++i) {
                    const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0;
                    result->set(i, src_ptr[idx]);
                }
                return py::cast(result);
            }
            throw py::type_error("Unsupported dtype for 1D asarray");
        }

        if (buf.ndim == 2) {
            if (is_numpy_float16(array)) {
                return py::cast(matrix_from_numpy(array));
            }
            if (py::isinstance<py::array_t<double>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<double>(array.cast<py::array_t<double>>()));
            }
            if (py::isinstance<py::array_t<float>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<float>(array.cast<py::array_t<float>>()));
            }
            if (py::isinstance<py::array_t<int32_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<int32_t>(array.cast<py::array_t<int32_t>>()));
            }
            if (py::isinstance<py::array_t<int64_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<int64_t>(array.cast<py::array_t<int64_t>>()));
            }
            if (py::isinstance<py::array_t<int8_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<int8_t>(array.cast<py::array_t<int8_t>>()));
            }
            if (py::isinstance<py::array_t<int16_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<int16_t>(array.cast<py::array_t<int16_t>>()));
            }
            if (py::isinstance<py::array_t<uint8_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<uint8_t>(array.cast<py::array_t<uint8_t>>()));
            }
            if (py::isinstance<py::array_t<uint16_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<uint16_t>(array.cast<py::array_t<uint16_t>>()));
            }
            if (py::isinstance<py::array_t<uint32_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<uint32_t>(array.cast<py::array_t<uint32_t>>()));
            }
            if (py::isinstance<py::array_t<uint64_t>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<uint64_t>(array.cast<py::array_t<uint64_t>>()));
            }
            if (py::isinstance<py::array_t<bool>>(array)) {
                auto tmp = array.cast<py::array_t<bool>>();
                auto b = tmp.request();
                require_2d(b);
                uint64_t rows = static_cast<uint64_t>(b.shape[0]);
                uint64_t cols = static_cast<uint64_t>(b.shape[1]);
                auto result = std::make_shared<DenseBitMatrix>(rows, cols);
                const bool* src_ptr = static_cast<const bool*>(b.ptr);
                const ptrdiff_t stride0 = static_cast<ptrdiff_t>(b.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
                const ptrdiff_t stride1 = static_cast<ptrdiff_t>(b.strides[1] / static_cast<py::ssize_t>(sizeof(bool)));
                
                uint64_t* dst_data = result->data();
                uint64_t dst_stride_bytes = result->stride_bytes();
                uint64_t dst_stride_words = dst_stride_bytes / 8;
                
                if (dst_data) {
                    for (uint64_t i = 0; i < rows; ++i) {
                        uint64_t* row_ptr = dst_data + i * dst_stride_words;
                        const bool* src_row = src_ptr + static_cast<ptrdiff_t>(i) * stride0;
                        
                        for (uint64_t j = 0; j < cols; j += 64) {
                            uint64_t word = 0;
                            uint64_t limit = std::min(cols - j, (uint64_t)64);
                            
                            if (stride1 == 1) {
                                const bool* src_chunk = src_row + j;
                                uint64_t k = 0;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                                __m128i zero = _mm_setzero_si128();
                                while (k + 16 <= limit) {
                                    __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_chunk + k));
                                    __m128i cmp = _mm_cmpeq_epi8(chunk, zero);
                                    int mask = _mm_movemask_epi8(cmp);
                                    uint64_t bits = static_cast<uint16_t>(~mask);
                                    word |= (bits << k);
                                    k += 16;
                                }
#endif
                                for (; k < limit; ++k) {
                                    if (src_chunk[k]) {
                                        word |= (1ULL << k);
                                    }
                                }
                            } else {
                                for (uint64_t k = 0; k < limit; ++k) {
                                    if (src_row[(j + k) * stride1]) {
                                        word |= (1ULL << k);
                                    }
                                }
                            }
                            row_ptr[j / 64] = word;
                        }
                    }
                    return py::cast(result);
                }

                for (uint64_t i = 0; i < rows; ++i) {
                    for (uint64_t j = 0; j < cols; ++j) {
                        const ptrdiff_t idx = static_cast<ptrdiff_t>(i) * stride0 + static_cast<ptrdiff_t>(j) * stride1;
                        result->set(i, j, src_ptr[idx]);
                    }
                }
                return py::cast(result);
            }
            if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<std::complex<float>>(array.cast<py::array_t<std::complex<float>>>()));
            }
            if (py::isinstance<py::array_t<std::complex<double>>>(array)) {
                return py::cast(dense_matrix_from_numpy_2d<std::complex<double>>(array.cast<py::array_t<std::complex<double>>>()));
            }
            throw py::type_error("Unsupported dtype for 2D asarray");
        }

        throw py::value_error("asarray only supports 1D or 2D arrays");
    });


    m.def(
        "cholesky",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::cholesky(*a);
                return std::shared_ptr<MatrixBase>(out.release());
            });
        },
        py::arg("a"));

    m.def(
        "eig",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eig(*a);
                return std::make_pair(
                    std::shared_ptr<VectorBase>(out.first.release()),
                    std::shared_ptr<MatrixBase>(out.second.release())
                );
            });
        },
        py::arg("a"));

    m.def(
        "eigvals",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigvals(*a);
                return std::shared_ptr<VectorBase>(out.release());
            });
        },
        py::arg("a"));

    m.def(
        "eigh",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigh(*a);
                return std::make_pair(
                    std::shared_ptr<VectorBase>(out.first.release()),
                    std::shared_ptr<MatrixBase>(out.second.release())
                );
            });
        },
        py::arg("a"));

    m.def(
        "eigvalsh",
        [](std::shared_ptr<MatrixBase> a) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigvalsh(*a);
                return std::shared_ptr<VectorBase>(out.release());
            });
        },
        py::arg("a"));

    m.def(
        "eigvals_arnoldi",
        [](std::shared_ptr<MatrixBase> a, int k, int m, double tol) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigvals_arnoldi(*a, k, m, tol);
                return std::shared_ptr<VectorBase>(out.release());
            });
        },
        py::arg("a"),
        py::arg("k"),
        py::arg("m"),
        py::arg("tol") = 1e-6);

    m.def(
        "eigvals_skew",
        [](std::shared_ptr<MatrixBase> a, int k) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eigvals_skew(*a, k);
                return std::shared_ptr<VectorBase>(out.release());
            });
        },
        py::arg("a"),
        py::arg("k"));

    m.def(
        "eig_skew",
        [](std::shared_ptr<MatrixBase> a, int k) {
            return translate_invalid_argument([&]() {
                auto out = pycauset::eig_skew(*a, k);
                return std::make_pair(
                    std::shared_ptr<VectorBase>(out.first.release()),
                    std::shared_ptr<MatrixBase>(out.second.release())
                );
            });
        },
        py::arg("a"),
        py::arg("k"));

    // The matrix inverse (pycauset.inv). NumPy's `invert` is bitwise NOT, so there is no
    // .invert() / .inverse() method (review 3.8 C-7).
    m.def("_inv", [](const MatrixBase& mat) { return matrix_inverse(mat); }, py::arg("a"));

    m.def(
        "_compute_k_matrix",
        [](std::shared_ptr<MatrixBase> C, double a) {
            return translate_invalid_argument(
                [&] { return std::shared_ptr<MatrixBase>(::compute_k_matrix(*C, a).release()); });
        },
        py::arg("C"),
        py::arg("a"));

}
