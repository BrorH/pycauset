#include "bindings_common.hpp"
#include "element_conversion.hpp"
#include "pycauset/math/Reductions.hpp"
#include "pycauset/core/ScalarUtils.hpp"

#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/core/AccessCounter.hpp"
#include "pycauset/core/MemoryHints.hpp"
#include "pycauset/core/IOAccelerator.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

namespace {

inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline pycauset::DataType parse_dtype(const std::string& s_raw) {
    const std::string s = to_lower(s_raw);
    if (s == "float16" || s == "f16" || s == "half") return pycauset::DataType::FLOAT16;
    if (s == "float32" || s == "f32") return pycauset::DataType::FLOAT32;
    if (s == "float64" || s == "f64" || s == "float") return pycauset::DataType::FLOAT64;
    if (s == "complex_float16" || s == "complex16") return pycauset::DataType::COMPLEX_FLOAT16;
    if (s == "complex_float32" || s == "complex64") return pycauset::DataType::COMPLEX_FLOAT32;
    if (s == "complex_float64" || s == "complex128" || s == "complex") return pycauset::DataType::COMPLEX_FLOAT64;
    if (s == "int8" || s == "i8") return pycauset::DataType::INT8;
    if (s == "int32" || s == "i32" || s == "int") return pycauset::DataType::INT32;
    if (s == "int16" || s == "i16") return pycauset::DataType::INT16;
    if (s == "int64" || s == "i64") return pycauset::DataType::INT64;
    if (s == "uint8" || s == "u8") return pycauset::DataType::UINT8;
    if (s == "uint16" || s == "u16") return pycauset::DataType::UINT16;
    if (s == "uint32" || s == "u32" || s == "uint") return pycauset::DataType::UINT32;
    if (s == "uint64" || s == "u64") return pycauset::DataType::UINT64;
    if (s == "bit" || s == "bool") return pycauset::DataType::BIT;
    throw std::invalid_argument("Unknown dtype: " + s_raw);
}

inline std::string dtype_to_string(pycauset::DataType dt) {
    switch (dt) {
        case pycauset::DataType::FLOAT16: return "float16";
        case pycauset::DataType::FLOAT32: return "float32";
        case pycauset::DataType::FLOAT64: return "float64";
        case pycauset::DataType::COMPLEX_FLOAT16: return "complex_float16";
        case pycauset::DataType::COMPLEX_FLOAT32: return "complex_float32";
        case pycauset::DataType::COMPLEX_FLOAT64: return "complex_float64";
        case pycauset::DataType::INT8: return "int8";
        case pycauset::DataType::INT32: return "int32";
        case pycauset::DataType::INT16: return "int16";
        case pycauset::DataType::INT64: return "int64";
        case pycauset::DataType::UINT8: return "uint8";
        case pycauset::DataType::UINT16: return "uint16";
        case pycauset::DataType::UINT32: return "uint32";
        case pycauset::DataType::UINT64: return "uint64";
        case pycauset::DataType::BIT: return "bit";
        default: return "unknown";
    }
}

inline pycauset::promotion::BinaryOp parse_op(const std::string& s_raw) {
    const std::string s = to_lower(s_raw);
    if (s == "add") return pycauset::promotion::BinaryOp::Add;
    if (s == "subtract" || s == "sub") return pycauset::promotion::BinaryOp::Subtract;
    if (s == "elementwise_multiply" || s == "elem_mul" || s == "mul") return pycauset::promotion::BinaryOp::ElementwiseMultiply;
    if (s == "divide" || s == "truediv" || s == "elementwise_divide" || s == "div") return pycauset::promotion::BinaryOp::Divide;
    if (s == "matmul") return pycauset::promotion::BinaryOp::Matmul;
    if (s == "matvec" || s == "matrix_vector_multiply") return pycauset::promotion::BinaryOp::MatrixVectorMultiply;
    if (s == "vecmat" || s == "vector_matrix_multiply") return pycauset::promotion::BinaryOp::VectorMatrixMultiply;
    throw std::invalid_argument("Unknown op: " + s_raw);
}

}

void bind_core_classes(py::module_& m) {
    m.def("set_num_threads", [](size_t n) { pycauset::ThreadPool::set_num_threads(n); }, py::arg("n"));
    m.def("get_num_threads", []() { return pycauset::ThreadPool::get_num_threads(); });

    m.def("set_memory_threshold", &pycauset::set_memory_threshold, py::arg("bytes"));
    m.def("get_memory_threshold", &pycauset::get_memory_threshold);

    // Storage root for auto-created backing files (.tmp). Public API lives in Python:
    // pycauset.set_backing_dir(...)
    m.def(
        "set_storage_root",
        [](const std::string& path) { pycauset::set_storage_root(std::filesystem::path(path)); },
        py::arg("path"));
    m.def(
        "get_storage_root",
        []() { return pycauset::get_storage_root().string(); });


    m.def(
        "norm",
        [](const pycauset::VectorBase& v) { return pycauset::norm(v); },
        py::arg("x"));

    m.def(
        "norm",
        [](const pycauset::MatrixBase& mat) { return pycauset::norm(mat); },
        py::arg("x"));

    m.def("sum", [](const pycauset::VectorBase& v) { return pycauset_bindings::reduction_to_py(pycauset::sum(v)); },
          py::arg("x"));
    m.def("sum", [](const pycauset::MatrixBase& mat) { return pycauset_bindings::reduction_to_py(pycauset::sum(mat)); },
          py::arg("x"));

    // --- Internal / test-only helpers ---
    // Exposes the promotion resolver without running kernels.
    m.def(
        "_debug_resolve_promotion",
        [](const std::string& op, const std::string& a_dtype, const std::string& b_dtype) {
            auto decision = pycauset::promotion::resolve(parse_op(op), parse_dtype(a_dtype), parse_dtype(b_dtype));
            py::dict out;
            out["result_dtype"] = dtype_to_string(decision.result_dtype);
            out["float_underpromotion"] = decision.float_underpromotion;
            out["chosen_float_dtype"] = dtype_to_string(decision.chosen_float_dtype);
            return out;
        },
        py::arg("op"),
        py::arg("a_dtype"),
        py::arg("b_dtype"));

    // Test-only: the governor's RAM safety margin. A huge margin makes every operation
    // "not fit in RAM", forcing the out-of-core StreamingManager path (F-026).
    m.def("_get_ram_safety_margin", []() { return pycauset::core::MemoryGovernor::instance().get_safety_margin(); });
    // Lazy results on/off (LAZY.md section 10: the lazy == eager test oracle).
    m.def("_set_lazy", [](bool on) { pycauset::lazy_enabled().store(on); }, py::arg("on"));
    m.def("_get_lazy", []() { return pycauset::lazy_enabled().load(); });

    m.def(
        "_set_ram_safety_margin",
        [](uint64_t bytes) { pycauset::core::MemoryGovernor::instance().set_safety_margin(bytes); },
        py::arg("bytes"));

    m.def("_debug_last_kernel_trace", []() { return pycauset::debug_trace::get_last(); });
    m.def("_debug_clear_kernel_trace", []() { pycauset::debug_trace::clear(); });
    m.def("_debug_kernel_log_start", []() { pycauset::debug_trace::start_log(); });
    m.def("_debug_kernel_log_stop", []() { return pycauset::debug_trace::stop_log(); });
    // Virtual per-element accessor counter (operation atlas): start resets and enables,
    // stop disables and returns the count since start.
    m.def("_debug_access_count_start", []() {
        pycauset::access_counter::g_count.store(0);
        pycauset::access_counter::g_enabled.store(true);
    });
    m.def("_debug_access_count_stop", []() {
        pycauset::access_counter::g_enabled.store(false);
        return pycauset::access_counter::g_count.load();
    });

    // --- Lookahead protocol / IO accelerator (Phase F) ---
    py::enum_<pycauset::core::AccessPattern>(m, "AccessPattern")
        .value("Sequential", pycauset::core::AccessPattern::Sequential)
        .value("Reverse", pycauset::core::AccessPattern::Reverse)
        .value("Strided", pycauset::core::AccessPattern::Strided)
        .value("Random", pycauset::core::AccessPattern::Random)
        .value("Once", pycauset::core::AccessPattern::Once);

    py::class_<pycauset::core::MemoryHint>(m, "MemoryHint")
        .def(py::init<>())
        .def_readwrite("pattern", &pycauset::core::MemoryHint::pattern)
        .def_readwrite("start_offset", &pycauset::core::MemoryHint::start_offset)
        .def_readwrite("length", &pycauset::core::MemoryHint::length)
        .def_readwrite("stride_bytes", &pycauset::core::MemoryHint::stride_bytes)
        .def_readwrite("block_bytes", &pycauset::core::MemoryHint::block_bytes)
        .def_static("sequential", &pycauset::core::MemoryHint::sequential, py::arg("start"), py::arg("len"))
        .def_static(
            "strided",
            &pycauset::core::MemoryHint::strided,
            py::arg("start"),
            py::arg("len"),
            py::arg("stride"),
            py::arg("block"));

    py::class_<pycauset::core::IOAccelerator>(m, "IOAccelerator")
        .def("prefetch", &pycauset::core::IOAccelerator::prefetch, py::arg("offset"), py::arg("size"))
        .def("discard", &pycauset::core::IOAccelerator::discard, py::arg("offset"), py::arg("size"))
        .def("process_hint", &pycauset::core::IOAccelerator::process_hint, py::arg("hint"));

    m.def("_debug_last_io_trace", []() { return pycauset::debug_trace::get_last_io(); });
    m.def("_debug_clear_io_trace", []() { pycauset::debug_trace::clear_io(); });
}
