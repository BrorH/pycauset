import time
import numpy as np
import pycauset
import sys

def benchmark_suite():
    if not pycauset.cuda.is_available():
        print("CUDA not available. Skipping benchmarks.")
        return

    print("\n" + "="*80)
    print("PYCAUSET GPU BENCHMARK SUITE")
    print("="*80)
    
    # 1. Matrix Multiplication
    print("\n[Benchmark 1: Matrix Multiplication (Float64)]")
    sizes = [1000, 2000, 4000]
    
    print(f"{'Size (NxN)':<12} | {'Sync (s)':<12} | {'Async (s)':<12} | {'Speedup':<10}")
    print("-" * 55)
    
    for N in sizes:
        # Generate Data
        A_np = np.random.rand(N, N).astype(np.float64)
        B_np = np.random.rand(N, N).astype(np.float64)
        A = pycauset.asarray(A_np)
        B = pycauset.asarray(B_np)
        
        # Force Tiling by setting low memory limit?
        # 4000x4000 doubles = 128MB.
        # If we want to force tiling for 2000 (32MB), we need limit < 100MB.
        limit = 100 * 1024 * 1024 
        
        # Sync
        pycauset.cuda.enable(memory_limit=limit, enable_async=False)
        _ = A @ B # Warmup
        start = time.time()
        _ = A @ B
        sync_time = time.time() - start
        
        # Async
        pycauset.cuda.enable(memory_limit=limit, enable_async=True)
        _ = A @ B # Warmup
        start = time.time()
        _ = A @ B
        async_time = time.time() - start
        
        speedup = sync_time / async_time if async_time > 0 else 0
        print(f"{N:<12} | {sync_time:<12.4f} | {async_time:<12.4f} | {speedup:<10.2f}x")

    # 2. Eigenvalues (Arnoldi)
    print("\n[Benchmark 2: Eigenvalues (Arnoldi, k=10)]")
    sizes = [1000, 2000]
    print(f"{'Size (NxN)':<12} | {'Time (s)':<12}")
    print("-" * 30)
    
    for N in sizes:
        A_np = np.random.rand(N, N).astype(np.float64)
        A = pycauset.asarray(A_np)
        
        pycauset.cuda.enable(enable_async=True)
        
        start = time.time()
        _ = pycauset.eigvals(A, k=10, method="arnoldi")
        duration = time.time() - start
        
        print(f"{N:<12} | {duration:<12.4f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    benchmark_suite()
