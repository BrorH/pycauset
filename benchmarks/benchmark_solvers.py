import time
import numpy as np
import pycauset
import os

def benchmark_inverse(n, dtype, iterations=3):
    print(f"\n--- Benchmarking Inverse (N={n}, {dtype.__name__}) ---")
    
    # Generate random matrix
    a_np = np.random.rand(n, n).astype(dtype)
    # Make it diagonally dominant to ensure invertibility
    a_np += np.eye(n, dtype=dtype) * n 
    
    # PyCauset
    a_pc = pycauset.asarray(a_np)
    
    # Warmup
    try:
        _ = a_pc.inverse()
    except Exception as e:
        print(f"PyCauset Warmup failed: {e}")
        return

    start = time.perf_counter()
    for _ in range(iterations):
        res = a_pc.inverse()
    end = time.perf_counter()
    pc_time = (end - start) / iterations
    print(f"PyCauset (GPU): {pc_time:.4f} s")
    
    # NumPy (CPU)
    start = time.perf_counter()
    for _ in range(iterations):
        res_np = np.linalg.inv(a_np)
    end = time.perf_counter()
    np_time = (end - start) / iterations
    print(f"NumPy (CPU):    {np_time:.4f} s")
    
    speedup = np_time / pc_time if pc_time > 0 else 0
    print(f"Speedup:        {speedup:.2f}x")

def benchmark_eigvals(n, dtype, iterations=3):
    print(f"\n--- Benchmarking Eigenvalues (Dense) (N={n}, {dtype.__name__}) ---")
    
    a_np = np.random.rand(n, n).astype(dtype)
    a_pc = pycauset.asarray(a_np)
    
    # Warmup
    try:
        _ = pycauset.eigvals(a_pc)
    except Exception as e:
        print(f"PyCauset Warmup failed: {e}")
        return

    start = time.perf_counter()
    for _ in range(iterations):
        res = pycauset.eigvals(a_pc)
    end = time.perf_counter()
    pc_time = (end - start) / iterations
    print(f"PyCauset (GPU): {pc_time:.4f} s")
    
    # NumPy (CPU)
    start = time.perf_counter()
    for _ in range(iterations):
        res_np = np.linalg.eigvals(a_np)
    end = time.perf_counter()
    np_time = (end - start) / iterations
    print(f"NumPy (CPU):    {np_time:.4f} s")
    
    speedup = np_time / pc_time if pc_time > 0 else 0
    print(f"Speedup:        {speedup:.2f}x")

def benchmark_arnoldi(n, dtype, k=10, iterations=3):
    print(f"\n--- Benchmarking Eigenvalues (Arnoldi) (N={n}, k={k}, {dtype.__name__}) ---")
    
    a_np = np.random.rand(n, n).astype(dtype)
    a_pc = pycauset.asarray(a_np)
    
    # Warmup
    try:
        _ = pycauset.eigvals_arnoldi(a_pc, k, max_iter=k*3)
    except Exception as e:
        print(f"PyCauset Warmup failed: {e}")
        return

    start = time.perf_counter()
    for _ in range(iterations):
        res = pycauset.eigvals_arnoldi(a_pc, k, max_iter=k*3)
    end = time.perf_counter()
    pc_time = (end - start) / iterations
    print(f"PyCauset (GPU+CPU): {pc_time:.4f} s")
    
    # Scipy (CPU) - if available
    try:
        import scipy.sparse.linalg
        start = time.perf_counter()
        for _ in range(iterations):
            res_sp = scipy.sparse.linalg.eigs(a_np, k=k)
        end = time.perf_counter()
        sp_time = (end - start) / iterations
        print(f"SciPy (CPU):        {sp_time:.4f} s")
        
        speedup = sp_time / pc_time if pc_time > 0 else 0
        print(f"Speedup:            {speedup:.2f}x")
    except ImportError:
        print("SciPy not installed, skipping CPU comparison.")

if __name__ == "__main__":
    print("Starting Solvers Benchmark...")
    
    # Test Sizes
    sizes = [500, 1000, 2000]
    
    # 1. Float32 (Consumer GPU Optimized)
    print("\n========================================")
    print("       FLOAT32 BENCHMARKS")
    print("========================================")
    for n in sizes:
        benchmark_inverse(n, np.float32)
        benchmark_eigvals(n, np.float32)
        benchmark_arnoldi(n, np.float32, k=20)

    # 2. Float64 (High Precision)
    print("\n========================================")
    print("       FLOAT64 BENCHMARKS")
    print("========================================")
    for n in sizes:
        benchmark_inverse(n, np.float64)
        benchmark_eigvals(n, np.float64)
        # Arnoldi is less sensitive to precision for timing, but good to check
        benchmark_arnoldi(n, np.float64, k=20)
