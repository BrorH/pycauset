#pragma once

#include "pycauset/core/StorageUtils.hpp"
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
struct MatrixTraits<int32_t> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::INT32;
    static constexpr const char* name = "int32";
};

template <>
struct MatrixTraits<bool> {
    static constexpr pycauset::DataType data_type = pycauset::DataType::BIT;
    static constexpr const char* name = "bit";
};

