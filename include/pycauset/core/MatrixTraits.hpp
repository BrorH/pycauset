#pragma once

#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/Float16.hpp"
#include <complex>
#include <cstdint>

template <typename T>
struct MatrixTraits;

template <>
struct MatrixTraits<double> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::FLOAT64;
    static constexpr const char* name = "float";
};

template <>
struct MatrixTraits<float> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::FLOAT32;
    static constexpr const char* name = "float32";
};

template <>
struct MatrixTraits<pycauset::float16_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::FLOAT16;
    static constexpr const char* name = "float16";
};

template <>
struct MatrixTraits<std::complex<float>> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::COMPLEX_FLOAT32;
    static constexpr const char* name = "complex_float32";
};

template <>
struct MatrixTraits<std::complex<double>> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::COMPLEX_FLOAT64;
    static constexpr const char* name = "complex_float64";
};



template <>
struct MatrixTraits<int32_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::INT32;
    static constexpr const char* name = "int32";
};

template <>
struct MatrixTraits<int16_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::INT16;
    static constexpr const char* name = "int16";
};

template <>
struct MatrixTraits<int8_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::INT8;
    static constexpr const char* name = "int8";
};

template <>
struct MatrixTraits<int64_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::INT64;
    static constexpr const char* name = "int64";
};

template <>
struct MatrixTraits<uint8_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::UINT8;
    static constexpr const char* name = "uint8";
};

template <>
struct MatrixTraits<uint16_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::UINT16;
    static constexpr const char* name = "uint16";
};

template <>
struct MatrixTraits<uint32_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::UINT32;
    static constexpr const char* name = "uint32";
};

template <>
struct MatrixTraits<uint64_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::UINT64;
    static constexpr const char* name = "uint64";
};

template <>
struct MatrixTraits<bool> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::BIT;
    static constexpr const char* name = "bit";
};

