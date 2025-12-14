#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/core/ObjectFactory.hpp"

using namespace pycauset;

void test_symmetric_access() {
    std::cout << "Testing Symmetric Access..." << std::endl;
    SymmetricMatrix<double> mat(10, "", false); // Symmetric

    mat.set(2, 5, 3.14);
    assert(std::abs(mat.get(2, 5) - 3.14) < 1e-9);
    assert(std::abs(mat.get(5, 2) - 3.14) < 1e-9); // Symmetric access

    mat.set(5, 2, 1.23); // Overwrite via swapped index
    assert(std::abs(mat.get(2, 5) - 1.23) < 1e-9);
    assert(std::abs(mat.get(5, 2) - 1.23) < 1e-9);

    std::cout << "  Passed." << std::endl;
}

void test_antisymmetric_access() {
    std::cout << "Testing Anti-Symmetric Access..." << std::endl;
    SymmetricMatrix<double> mat(10, "", true); // Anti-Symmetric

    mat.set(2, 5, 2.0);
    assert(std::abs(mat.get(2, 5) - 2.0) < 1e-9);
    assert(std::abs(mat.get(5, 2) - (-2.0)) < 1e-9); // Negated access

    mat.set(5, 2, 3.0); // Set lower: A[5,2]=3 => A[2,5]=-3
    assert(std::abs(mat.get(2, 5) - (-3.0)) < 1e-9);
    assert(std::abs(mat.get(5, 2) - 3.0) < 1e-9);

    // Diagonal check
    try {
        mat.set(3, 3, 1.0);
        assert(false && "Should have thrown exception for non-zero diagonal");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    mat.set(3, 3, 0.0); // Should be fine
    assert(std::abs(mat.get(3, 3)) < 1e-9);

    std::cout << "  Passed." << std::endl;
}

void test_persistence() {
    std::cout << "Testing Persistence..." << std::endl;
    std::string filename = "test_sym.pycauset";
    
    {
        SymmetricMatrix<double> mat(10, filename, true); // Anti-Symmetric
        mat.set(1, 2, 42.0);
        mat.set_temporary(false); // Keep file
    }

    // Load back
    auto loaded = ObjectFactory::load_matrix(filename, 0, 10, 10, DataType::FLOAT64, MatrixType::ANTISYMMETRIC);
    auto* sym = dynamic_cast<SymmetricMatrix<double>*>(loaded.get());
    
    assert(sym != nullptr);
    assert(sym->is_antisymmetric());
    assert(std::abs(sym->get(1, 2) - 42.0) < 1e-9);
    assert(std::abs(sym->get(2, 1) - (-42.0)) < 1e-9);

    loaded->close();
    std::filesystem::remove(filename);
    std::cout << "  Passed." << std::endl;
}

void test_clone() {
    std::cout << "Testing Clone..." << std::endl;
    SymmetricMatrix<double> mat(10, "", true);
    mat.set(1, 2, 10.0);

    auto clone = mat.clone();
    auto* sym_clone = dynamic_cast<SymmetricMatrix<double>*>(clone.get());

    assert(sym_clone != nullptr);
    assert(sym_clone->is_antisymmetric());
    assert(std::abs(sym_clone->get(1, 2) - 10.0) < 1e-9);

    std::cout << "  Passed." << std::endl;
}

void test_antisymmetric_class() {
    std::cout << "Testing AntiSymmetricMatrix Class..." << std::endl;
    AntiSymmetricMatrix<double> mat(10);

    assert(mat.is_antisymmetric());
    
    mat.set(2, 5, 2.0);
    assert(std::abs(mat.get(2, 5) - 2.0) < 1e-9);
    assert(std::abs(mat.get(5, 2) - (-2.0)) < 1e-9);

    auto clone = mat.clone();
    auto* clone_ptr = dynamic_cast<AntiSymmetricMatrix<double>*>(clone.get());
    assert(clone_ptr != nullptr);
    assert(clone_ptr->is_antisymmetric());

    std::cout << "  Passed." << std::endl;
}

int main() {
    try {
        test_symmetric_access();
        test_antisymmetric_access();
        test_antisymmetric_class();
        test_persistence();
        test_clone();
        std::cout << "All SymmetricMatrix tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test Failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}