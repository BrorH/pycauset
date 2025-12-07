#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <functional>
#include <filesystem>
#include <bit>

#include "DenseMatrix.hpp"
#include "DenseBitMatrix.hpp"
#include "DenseVector.hpp"
#include "TriangularMatrix.hpp"
#include "ComputeContext.hpp"
#include "MatrixOperations.hpp"
#include "VectorOperations.hpp"

using namespace pycauset;

// Timer helper
class Timer {
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer(const std::string& n) : name(n), start(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        std::cout << std::left << std::setw(40) << name << ": " << ms.count() << " ms" << std::endl;
    }
};

std::string temp_file(const std::string& name) {
    return "bench_temp_" + name + ".bin";
}

void benchmark_bit_vector_ops(int N) {
    std::cout << "\n--- Bit Vector Operations (N=" << N << ") ---" << std::endl;
    
    auto v1 = std::make_unique<DenseVector<bool>>(N, temp_file("v1"));
    auto v2 = std::make_unique<DenseVector<bool>>(N, temp_file("v2"));
    
    // Fill with random bits
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 1);
    for(int i=0; i<N; ++i) {
        v1->set(i, dis(gen));
        v2->set(i, dis(gen));
    }
    
    // Naive Dot Product
    {
        Timer t("Naive Dot Product (Loop)");
        int sum = 0;
        for(int i=0; i<N; ++i) {
            if (v1->get(i) && v2->get(i)) sum++;
        }
        volatile int keep = sum;
    }
    
    // Optimized Dot Product
    {
        Timer t("Optimized Dot Product (Popcount)");
        double res = dot_product(*v1, *v2);
        volatile double keep = res;
    }
}

void benchmark_bit_matrix_vector(int N) {
    std::cout << "\n--- Bit Matrix-Vector Multiply (N=" << N << ") ---" << std::endl;
    
    auto M = std::make_unique<DenseMatrix<bool>>(N, temp_file("M"));
    auto v = std::make_unique<DenseVector<bool>>(N, temp_file("v"));
    
    // Fill
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 1);
    for(int i=0; i<N; ++i) {
        v->set(i, dis(gen));
        for(int j=0; j<N; ++j) {
            M->set(i, j, dis(gen));
        }
    }
    
    // Naive
    {
        Timer t("Naive Matrix-Vector");
        std::vector<int> res(N, 0);
        for(int i=0; i<N; ++i) {
            for(int j=0; j<N; ++j) {
                if (M->get(i, j) && v->get(j)) res[i]++;
            }
        }
    }
    
    // Optimized
    {
        Timer t("Optimized Matrix-Vector (Popcount)");
        auto res = matrix_vector_multiply(*M, *v, temp_file("res"));
    }
}

void benchmark_triangular_inverse(int N) {
    std::cout << "\n--- Triangular Matrix Inverse (N=" << N << ") ---" << std::endl;
    
    auto T = std::make_unique<TriangularMatrix<double>>(N, temp_file("T"), true); // Has diagonal
    
    // Fill
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.1, 1.0);
    for(int i=0; i<N; ++i) {
        for(int j=i; j<N; ++j) {
            T->set(i, j, dis(gen));
        }
    }
    
    auto Inv = std::make_unique<TriangularMatrix<double>>(N, temp_file("Inv"), true);
    
    // Optimized
    {
        Timer t("Optimized Triangular Inverse");
        ComputeContext::instance().get_device()->inverse(*T, *Inv);
    }
}

int main() {
    try {
        benchmark_bit_vector_ops(10000000); // 10M bits
        benchmark_bit_matrix_vector(2000);  // 2000x2000
        benchmark_triangular_inverse(1000); // 1000x1000
        
        // Cleanup
        for(const auto& entry : std::filesystem::directory_iterator(".")) {
            if (entry.path().string().find("bench_temp_") != std::string::npos) {
                std::filesystem::remove(entry.path());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
