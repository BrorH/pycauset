#pragma once
#include <memory>
#include <string>
#include "VectorBase.hpp"

namespace pycauset {

std::unique_ptr<VectorBase> add_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
std::unique_ptr<VectorBase> subtract_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
double dot_product(const VectorBase& a, const VectorBase& b);
std::unique_ptr<VectorBase> cross_product(const VectorBase& a, const VectorBase& b, const std::string& result_file = "");
std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, double scalar, const std::string& result_file = "");
std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, int64_t scalar, const std::string& result_file = "");
std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, double scalar, const std::string& result_file = "");
std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, int64_t scalar, const std::string& result_file = "");

}
