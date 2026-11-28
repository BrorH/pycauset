#include "bindings_common.hpp"
#include "element_conversion.hpp"
#include "pycauset/math/Reductions.hpp"
#include "pycauset/math/DerivedCache.hpp"
#include "pycauset/core/ScalarUtils.hpp"

#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/core/AccessCounter.hpp"
#include "pycauset/core/MemoryHints.hpp"
#include "pycauset/core/IOAccelerator.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
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
    m.def("set_num_threads", [](size_t n) { pycauset::set_num_threads(n); }, py::arg("n"));
    m.def("get_num_threads", []() { return pycauset::ThreadPool::get_num_threads(); });

    m.def("set_memory_threshold", &pycauset::set_memory_threshold, py::arg("bytes"));
    m.def("get_memory_threshold", &pycauset::get_memory_threshold);
    m.def("_set_thread_memory_threshold", &pycauset::set_thread_memory_threshold, py::arg("bytes"));

    // Storage root for auto-created backing files (.pycauset-tmp). Public API lives in Python:
    // pycauset.set_backing_dir(...)
    m.def(
        "_set_storage_root",  // internal: pycauset.set_backing_dir keeps Python and the engine in step
        [](const std::string& path) { pycauset::set_storage_root(std::filesystem::path(path)); },
        py::arg("path"));


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
    // Internal: `a` converted to `dtype` in the engine (a dense copy; the block layer casts its
    // leaves to a grid's result dtype with it, review 3.10 C-4).
    m.def("_cast", [](const pycauset::MatrixBase& a, const std::string& dtype) {
        const auto dt = parse_dtype(dtype);
        if (dt == pycauset::DataType::BIT) throw std::invalid_argument("_cast: no cast to bool");
        return std::shared_ptr<pycauset::MatrixBase>(pycauset::cast_to(a, dt).release());
    }, py::arg("a"), py::arg("dtype"));

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
    // "not fit in RAM": every new object is file-backed (tests/engine/test_out_of_core.py).
    m.def("_get_ram_safety_margin", []() { return pycauset::core::MemoryGovernor::instance().get_safety_margin(); });
    // Lazy results on/off (LAZY.md section 10: the lazy == eager test oracle).
    m.def("_set_lazy", [](bool on) { pycauset::lazy_enabled().store(on); }, py::arg("on"));
    m.def("_get_lazy", []() { return pycauset::lazy_enabled().load(); });
    // Derived-value cache (math/DerivedCache.hpp): counters and an off switch for tests and
    // for timing cold computations; the storage version of an object (moves on every write).
    m.def("_derived_cache_stats", []() {
        return py::make_tuple(pycauset::derived_cache::hits().load(), pycauset::derived_cache::stores().load());
    });
    m.def("_set_derived_cache", [](bool on) { pycauset::derived_cache::enabled().store(on); }, py::arg("on"));
    // Saving derived values with a file (DERIVED_CACHE.md D.3): the scalar entries of the
    // object's storage as [key, kind, values...] (vector entries stay in memory), and back.
    m.def("_derived_cache_export", [](const pycauset::MatrixBase& mat) {
        py::list out;
        if (!mat.shared_mapper()) return out;
        using pycauset::ReductionValue;
        for (const auto& [key, value] : mat.require_mapper()->cache_snapshot()) {
            if (const auto* c = std::any_cast<std::complex<double>>(&value)) {
                out.append(py::make_tuple(key, "complex", c->real(), c->imag()));
            } else if (const auto* sl = std::any_cast<std::pair<std::complex<double>, double>>(&value)) {
                out.append(py::make_tuple(key, "slogdet", sl->first.real(), sl->first.imag(), sl->second));
            } else if (const auto* d = std::any_cast<double>(&value)) {
                out.append(py::make_tuple(key, "real", *d));
            } else if (const auto* i = std::any_cast<int64_t>(&value)) {
                out.append(py::make_tuple(key, "int", *i));
            } else if (const auto* r = std::any_cast<ReductionValue>(&value)) {
                // The 128-bit integer as text: the container's integers are 64-bit.
                __int128 v = r->i;
                const bool neg = v < 0;
                unsigned __int128 u = neg ? static_cast<unsigned __int128>(-(v + 1)) + 1 : static_cast<unsigned __int128>(v);
                std::string digits;
                do { digits.insert(digits.begin(), static_cast<char>('0' + static_cast<int>(u % 10))); u /= 10; } while (u);
                if (neg) digits.insert(digits.begin(), '-');
                out.append(py::make_tuple(key, "reduction", static_cast<int>(r->kind), digits, r->r, r->c.real(), r->c.imag()));
            }
        }
        return out;
    });
    m.def("_derived_cache_import", [](const pycauset::MatrixBase& mat, const py::list& entries) {
        const auto& mapper = mat.shared_mapper();
        if (!mapper) return;
        const uint64_t version = mapper->version();
        for (const auto& item : entries) {
            const py::sequence e = py::reinterpret_borrow<py::sequence>(item);
            const std::string key = e[0].cast<std::string>();
            const std::string kind = e[1].cast<std::string>();
            std::any value;
            if (kind == "complex") value = std::complex<double>(e[2].cast<double>(), e[3].cast<double>());
            else if (kind == "slogdet")
                value = std::pair<std::complex<double>, double>({e[2].cast<double>(), e[3].cast<double>()}, e[4].cast<double>());
            else if (kind == "real") value = e[2].cast<double>();
            else if (kind == "int") value = e[2].cast<int64_t>();
            else if (kind == "reduction") {
                pycauset::ReductionValue r;
                r.kind = static_cast<pycauset::ReductionValue::Kind>(e[2].cast<int>());
                const std::string digits = e[3].cast<std::string>();
                __int128 v = 0;
                const bool neg = !digits.empty() && digits[0] == '-';
                for (size_t k = neg ? 1 : 0; k < digits.size(); ++k) v = v * 10 + (digits[k] - '0');
                r.i = neg ? -v : v;
                r.r = e[4].cast<double>();
                r.c = {e[5].cast<double>(), e[6].cast<double>()};
                value = r;
            } else {
                throw py::value_error("derived-value entry of unknown kind: " + kind);
            }
            mapper->cache_put(key, version, std::move(value));
        }
    }, py::arg("mat"), py::arg("entries"));
    m.def("_storage_version", [](const pycauset::MatrixBase& mat) -> uint64_t {
        return mat.shared_mapper() ? mat.shared_mapper()->version() : 0;
    });
    m.def("_storage_version", [](const pycauset::VectorBase& vec) -> uint64_t {
        return vec.shared_mapper() ? vec.shared_mapper()->version() : 0;
    });

    // pc.save's payload: the storage's bytes written straight into the container at `offset`
    // (review 3.9 C-2: it was copied into a populated staging file first, then copied again in
    // Python: twice the disk, the payload added to RSS, ~8x np.save's time). Returns the
    // number of bytes written. The GIL is released during the write.
    auto write_payload = [](const std::shared_ptr<MemoryMapper>& mapper, const std::string& path, uint64_t offset) {
        if (!mapper) throw std::invalid_argument("operation on a closed matrix or vector (close() released its storage)");
        const char* data = static_cast<const char*>(mapper->get_data());
        const uint64_t size = mapper->get_data_size();
        {
            py::gil_scoped_release release;
            std::fstream out(std::filesystem::path(reinterpret_cast<const char8_t*>(path.c_str())),
                             std::ios::in | std::ios::out | std::ios::binary);
            if (!out) throw std::runtime_error("save: cannot open " + path);
            out.seekp(static_cast<std::streamoff>(offset));
            constexpr uint64_t chunk = 64ull << 20;
            for (uint64_t done = 0; done < size && out; done += chunk) {
                out.write(data + done, static_cast<std::streamsize>(std::min(chunk, size - done)));
            }
            out.flush();
            if (!out) throw std::runtime_error("save: writing the payload to " + path + " failed");
        }
        return size;
    };
    // Bytes of storage an object maps (load checks it against the file's payload length).
    m.def("_storage_bytes", [](const pycauset::MatrixBase& mat) -> uint64_t {
        return mat.shared_mapper() ? mat.require_mapper()->get_data_size() : 0;
    });
    m.def("_storage_bytes", [](const pycauset::VectorBase& vec) -> uint64_t {
        return vec.shared_mapper() ? vec.require_mapper()->get_data_size() : 0;
    });
    m.def("_write_payload", [write_payload](const pycauset::MatrixBase& mat, const std::string& path, uint64_t offset) {
        return write_payload(mat.shared_mapper(), path, offset);
    }, py::arg("obj"), py::arg("path"), py::arg("offset"));
    m.def("_write_payload", [write_payload](const pycauset::VectorBase& vec, const std::string& path, uint64_t offset) {
        return write_payload(vec.shared_mapper(), path, offset);
    }, py::arg("obj"), py::arg("path"), py::arg("offset"));

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


    m.def("_debug_last_io_trace", []() { return pycauset::debug_trace::get_last_io(); });
    m.def("_debug_clear_io_trace", []() { pycauset::debug_trace::clear_io(); });
}
