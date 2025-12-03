#pragma once

#include "VectorBase.hpp"
#include "DenseVector.hpp"
#include "VectorOperations.hpp"
#include <complex>
#include <memory>
#include <string>

namespace pycauset {

using FloatVector = DenseVector<double>;

class ComplexVector {
public:
    // Create new complex vector (creates two backing files)
    ComplexVector(uint64_t n, const std::string& backing_file_real = "", const std::string& backing_file_imag = "") 
        : n_(n) {
        real_ = std::make_unique<FloatVector>(n, backing_file_real);
        imag_ = std::make_unique<FloatVector>(n, backing_file_imag);
    }

    // Construct from existing parts (takes ownership)
    ComplexVector(std::unique_ptr<FloatVector> real, std::unique_ptr<FloatVector> imag)
        : n_(real->size()), real_(std::move(real)), imag_(std::move(imag)) {
        if (n_ != imag_->size()) throw std::invalid_argument("Real and Imaginary parts must have same size");
    }

    uint64_t size() const { return n_; }

    std::complex<double> get(uint64_t i) const {
        return std::complex<double>(real_->get_element_as_double(i), imag_->get_element_as_double(i));
    }

    void set(uint64_t i, std::complex<double> val) {
        real_->set(i, val.real());
        imag_->set(i, val.imag());
    }

    FloatVector* real() { return real_.get(); }
    const FloatVector* real() const { return real_.get(); }
    
    FloatVector* imag() { return imag_.get(); }
    const FloatVector* imag() const { return imag_.get(); }

    // Conjugate
    std::unique_ptr<ComplexVector> conjugate(const std::string& saveas_real = "", const std::string& saveas_imag = "") const;

    void close() {
        real_->close();
        imag_->close();
    }

private:
    uint64_t n_;
    std::unique_ptr<FloatVector> real_;
    std::unique_ptr<FloatVector> imag_;
};

// Operations
std::unique_ptr<ComplexVector> add(const ComplexVector& a, const ComplexVector& b, const std::string& saveas_real = "", const std::string& saveas_imag = "");
std::unique_ptr<ComplexVector> subtract(const ComplexVector& a, const ComplexVector& b, const std::string& saveas_real = "", const std::string& saveas_imag = "");
std::complex<double> dot(const ComplexVector& a, const ComplexVector& b);
std::unique_ptr<ComplexVector> cross(const ComplexVector& a, const ComplexVector& b, const std::string& saveas_real = "", const std::string& saveas_imag = "");

std::unique_ptr<ComplexVector> multiply_scalar(const ComplexVector& v, std::complex<double> scalar, const std::string& saveas_real = "", const std::string& saveas_imag = "");
std::unique_ptr<ComplexVector> add_scalar(const ComplexVector& v, std::complex<double> scalar, const std::string& saveas_real = "", const std::string& saveas_imag = "");

}
