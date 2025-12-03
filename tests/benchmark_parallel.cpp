#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include "DenseMatrix.hpp"
#include "ParallelUtils.hpp"
#include "MatrixFactory.hpp"
#include "Eigen.hpp"

using namespace pycauset;

// Helper to measure execution time
template<typename Func>
double measure_time(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

void run_inversion_benchmark(size_t N) {
    std::cout << "\n--- Benchmarking Inversion (N=" << N << ") ---" << std::endl;
    std::string file_A = "bench_inv_A.bin";
    std::string file_Res = "bench_inv_Res.bin";

    if (std::filesystem::exists(file_A)) std::filesystem::remove(file_A);
    if (std::filesystem::exists(file_Res)) std::filesystem::remove(file_Res);

    double t_seq = 0.0;
    double t_par = 0.0;

    {
        DenseMatrix<double> A(N, file_A);
        // Create a diagonally dominant matrix to ensure invertibility
        ParallelFor(0, N, [&](size_t i) {
            for(size_t j=0; j<N; ++j) {
                if (i == j) A.set(i, j, (double)N + 1.0);
                else A.set(i, j, 1.0);
            }
        });

        // Sequential
        ThreadPool::set_num_threads(1);
        t_seq = measure_time([&]() {
            auto Res = A.inverse(file_Res);
        });
        std::cout << "Sequential Time: " << t_seq << " s" << std::endl;
        
        if (std::filesystem::exists(file_Res)) std::filesystem::remove(file_Res);

        // Parallel
        size_t max_threads = std::thread::hardware_concurrency();
        ThreadPool::set_num_threads(max_threads);
        t_par = measure_time([&]() {
            auto Res = A.inverse(file_Res);
        });
        std::cout << "Parallel Time (" << max_threads << " threads): " << t_par << " s" << std::endl;
    } // A destroyed here

    std::cout << "Speedup: " << t_seq / t_par << "x" << std::endl;

    if (std::filesystem::exists(file_A)) std::filesystem::remove(file_A);
    if (std::filesystem::exists(file_Res)) std::filesystem::remove(file_Res);
}

void run_eigen_benchmark(size_t N) {
    std::cout << "\n--- Benchmarking Eigenvalues (N=" << N << ") ---" << std::endl;
    std::string file_A = "bench_eig_A.bin";
    
    if (std::filesystem::exists(file_A)) std::filesystem::remove(file_A);

    double t_seq = 0.0;
    double t_par = 0.0;

    {
        DenseMatrix<double> A(N, file_A);
        // Symmetric matrix for real eigenvalues
        ParallelFor(0, N, [&](size_t i) {
            for(size_t j=0; j<N; ++j) {
                double val = std::cos(i) * std::sin(j);
                A.set(i, j, val);
                A.set(j, i, val);
            }
        });

        // Sequential
        ThreadPool::set_num_threads(1);
        t_seq = measure_time([&]() {
            auto ev = eigvals(A);
        });
        std::cout << "Sequential Time: " << t_seq << " s" << std::endl;
        
        // Clear cache to force recomputation
        A.clear_cached_eigenvalues();

        // Parallel
        size_t max_threads = std::thread::hardware_concurrency();
        ThreadPool::set_num_threads(max_threads);
        t_par = measure_time([&]() {
            auto ev = eigvals(A);
        });
        std::cout << "Parallel Time (" << max_threads << " threads): " << t_par << " s" << std::endl;
    }

    std::cout << "Speedup: " << t_seq / t_par << "x" << std::endl;

    if (std::filesystem::exists(file_A)) std::filesystem::remove(file_A);
}

int main() {
    std::cout << "Running Extensive Benchmarks..." << std::endl;
    std::cout << "Hardware Concurrency: " << std::thread::hardware_concurrency() << std::endl;

    // Inversion Benchmark (Skipped as requested)
    // run_inversion_benchmark(500);
    // run_inversion_benchmark(1000);
    // run_inversion_benchmark(2000);
    // run_inversion_benchmark(5000);

    // Eigenvalue Benchmark
    // Start small to verify Hessenberg Reduction speedup
    run_eigen_benchmark(100);
    run_eigen_benchmark(200);
    run_eigen_benchmark(500);
    run_eigen_benchmark(1000);
    
    // Larger tests
    // run_eigen_benchmark(2000);
    // run_eigen_benchmark(5000);

    return 0;
}
