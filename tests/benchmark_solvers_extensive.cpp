#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <functional>
#include <filesystem>

#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/compute/cpu/CpuSolver.hpp"

using namespace pycauset;

// Naive implementations for comparison
class NaiveSolver {
public:
    static void matmul_dense_naive(const DenseMatrix<double>& A, const DenseMatrix<double>& B, DenseMatrix<double>& C) {
        size_t n = A.size();
        const double* a_ptr = A.data();
        const double* b_ptr = B.data();
        double* c_ptr = C.data();

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0;
                for (size_t k = 0; k < n; ++k) {
                    sum += a_ptr[i * n + k] * b_ptr[k * n + j];
                }
                c_ptr[i * n + j] = sum;
            }
        }
    }

    static void matmul_bit_naive(const DenseBitMatrix& A, const DenseBitMatrix& B, DenseMatrix<int32_t>& C) {
        size_t n = A.size();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                int32_t sum = 0;
                for (size_t k = 0; k < n; ++k) {
                    if (A.get(i, k) && B.get(k, j)) {
                        sum++;
                    }
                }
                C.set(i, j, sum);
            }
        }
    }
};

// Timer utility
template<typename Func>
double measure_ms(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_header(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::setw(10) << "Size" << " | " 
              << std::setw(15) << "Optimized (ms)" << " | " 
              << std::setw(15) << "Naive (ms)" << " | " 
              << std::setw(10) << "Speedup" << "\n";
    std::cout << std::string(60, '-') << "\n";
}

void print_row(size_t size, double opt_ms, double naive_ms) {
    std::cout << std::setw(10) << size << " | " 
              << std::setw(15) << std::fixed << std::setprecision(2) << opt_ms << " | " 
              << std::setw(15) << naive_ms << " | " 
              << std::setw(9) << std::fixed << std::setprecision(2) << (naive_ms / opt_ms) << "x\n";
}

void benchmark_dense_matmul() {
    print_header("Dense Matrix Multiplication (Double)");
    std::vector<size_t> sizes = {100, 256, 512, 1024};

    for (size_t n : sizes) {
        std::string fA = "bench_dense_A_" + std::to_string(n) + ".tmp";
        std::string fB = "bench_dense_B_" + std::to_string(n) + ".tmp";
        std::string fC = "bench_dense_C_" + std::to_string(n) + ".tmp";
        
        double t_opt = 0.0;
        double t_naive = 0.0;

        {
            DenseMatrix<double> A(n, fA);
            DenseMatrix<double> B(n, fB);
            DenseMatrix<double> C_opt(n, fC);
            DenseMatrix<double> C_naive(n, ""); // Memory only

            // Fill with random data
            std::mt19937 gen(42);
            std::uniform_real_distribution<> dis(0.0, 1.0);
            for(size_t i=0; i<n*n; ++i) {
                A.data()[i] = dis(gen);
                B.data()[i] = dis(gen);
            }

            // Optimized
            t_opt = measure_ms([&]() {
                ComputeContext::instance().get_device()->matmul(A, B, C_opt);
            });

            // Naive
            t_naive = measure_ms([&]() {
                NaiveSolver::matmul_dense_naive(A, B, C_naive);
            });
        }

        print_row(n, t_opt, t_naive);

        // Cleanup
        std::filesystem::remove(fA);
        std::filesystem::remove(fB);
        std::filesystem::remove(fC);
    }
}

void benchmark_bit_matmul() {
    print_header("Bit Matrix Multiplication (Boolean/GF2)");
    std::vector<size_t> sizes = {128, 512, 1024, 2048};

    for (size_t n : sizes) {
        std::string fA = "bench_bit_A_" + std::to_string(n) + ".tmp";
        std::string fB = "bench_bit_B_" + std::to_string(n) + ".tmp";
        std::string fC = "bench_bit_C_" + std::to_string(n) + ".tmp";
        
        double t_opt = 0.0;
        double t_naive = 0.0;

        {
            DenseBitMatrix A(n, fA);
            DenseBitMatrix B(n, fB);
            DenseMatrix<int32_t> C_opt(n, fC);
            DenseMatrix<int32_t> C_naive(n, "");

            // Fill with random data
            std::mt19937 gen(42);
            std::bernoulli_distribution dis(0.5);
            for(size_t i=0; i<n; ++i) {
                for(size_t j=0; j<n; ++j) {
                    if(dis(gen)) A.set(i, j, true);
                    if(dis(gen)) B.set(i, j, true);
                }
            }

            // Optimized
            t_opt = measure_ms([&]() {
                ComputeContext::instance().get_device()->matmul(A, B, C_opt);
            });

            // Naive (Only run for smaller sizes as it is VERY slow)
            if (n <= 1024) {
                t_naive = measure_ms([&]() {
                    NaiveSolver::matmul_bit_naive(A, B, C_naive);
                });
            } else {
                t_naive = -1.0; // Skip
            }
        }

        if (t_naive > 0)
            print_row(n, t_opt, t_naive);
        else
            std::cout << std::setw(10) << n << " | " 
                      << std::setw(15) << std::fixed << std::setprecision(2) << t_opt << " | " 
                      << std::setw(15) << "Skipped" << " | " 
                      << std::setw(10) << "-" << "\n";

        std::filesystem::remove(fA);
        std::filesystem::remove(fB);
        std::filesystem::remove(fC);
    }
}

void benchmark_diagonal_matmul() {
    print_header("Diagonal Matrix Multiplication (vs Dense)");
    std::vector<size_t> sizes = {1000, 5000, 10000};

    for (size_t n : sizes) {
        std::string fA = "bench_diag_A_" + std::to_string(n) + ".tmp";
        std::string fB = "bench_diag_B_" + std::to_string(n) + ".tmp";
        std::string fC = "bench_diag_C_" + std::to_string(n) + ".tmp";
        
        double t_opt = 0.0;
        double t_naive = 0.0;

        {
            DiagonalMatrix<double> A(n, fA);
            DiagonalMatrix<double> B(n, fB);
            DiagonalMatrix<double> C_opt(n, fC);
            
            // Optimized (Diagonal * Diagonal -> Diagonal)
            t_opt = measure_ms([&]() {
                ComputeContext::instance().get_device()->matmul(A, B, C_opt);
            });

            // Naive (Treating as Dense O(N^3) - Simulated)
            // We won't actually run O(N^3) for 10000, it would take forever.
            // We'll estimate based on N^3 scaling from previous dense benchmark or just skip.
            // Instead, let's compare against a "Naive Loop" that iterates N^2 but checks indices.
            
            t_naive = measure_ms([&]() {
                // Simulate naive dense iteration
                for(size_t i=0; i<n; ++i) {
                    // Just do the diagonal part to be fair to memory, but iterate
                    double val = A.get(i, i) * B.get(i, i);
                    C_opt.set(i, i, val);
                }
            });
        }
        
        // Actually, the real comparison is O(N) vs O(N^3).
        // Let's just show the time.
        
        std::cout << std::setw(10) << n << " | " 
                  << std::setw(15) << std::fixed << std::setprecision(4) << t_opt << " | " 
                  << std::setw(15) << "N/A (O(N))" << " | " 
                  << std::setw(10) << "Huge" << "\n";

        std::filesystem::remove(fA);
        std::filesystem::remove(fB);
        std::filesystem::remove(fC);
    }
}

int main() {
    try {
        std::cout << "Starting Extensive Benchmarks...\n";
        std::cout << "CPU: " << std::thread::hardware_concurrency() << " threads available.\n";
        
        benchmark_dense_matmul();
        benchmark_bit_matmul();
        benchmark_diagonal_matmul();
        
        std::cout << "\nBenchmarks Completed.\n";
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
