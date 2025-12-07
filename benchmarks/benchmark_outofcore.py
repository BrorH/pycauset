import numpy as np
import pycauset
import time
import sys

def benchmark_outofcore():
    if not pycauset.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    print("\n" + "="*80)
    print("BENCHMARK: OUT-OF-CORE INVERSE")
    print("="*80)

    sizes = [1024, 2048, 4096]
    limit = 100 * 1024 * 1024 # 100 MB Limit (Force Out-of-Core for larger sizes)
    
    # Warmup
    print("Warming up...")
    A_warm = pycauset.Float64Matrix(512)
    for i in range(512): A_warm[i, i] = 1.0
    A_warm.inverse()

    results = []

    for N in sizes:
        print(f"\nBenchmarking Size: {N}x{N}")
        
        # Create Matrix
        A = pycauset.Float64Matrix(N)
        # Fill with random data (fast fill if possible, else loop)
        # We'll just leave it as zeros? No, singular.
        # Fill diagonal
        print("  Generating data...")
        # Use a simple pattern to fill fast
        # We can use set_data with a list if exposed, but it's not.
        # We'll use the fact that new matrix is zero, set diagonal to 1.
        # Actually, we need a general matrix to test the solver speed properly.
        # Sparse fill is fast.
        for i in range(N):
            A[i, i] = 2.0
            if i > 0: A[i, i-1] = -1.0
            if i < N-1: A[i, i+1] = -1.0
            
        pycauset.cuda.enable(memory_limit=limit, enable_async=True)
        
        print("  Running Inverse...")
        start_time = time.time()
        A.inverse()
        end_time = time.time()
        
        duration = end_time - start_time
        flops = (2.0/3.0) * (N**3) + (4.0/3.0) * (N**3) # LU + Inverse (approx 2N^3 total)
        gflops = (flops / duration) / 1e9
        
        print(f"  Time: {duration:.4f} s")
        print(f"  Performance: {gflops:.2f} GFLOPS")
        
        results.append((N, duration, gflops))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Size':<10} | {'Time (s)':<10} | {'GFLOPS':<10}")
    print("-" * 36)
    for N, t, g in results:
        print(f"{N:<10} | {t:<10.4f} | {g:<10.2f}")

if __name__ == "__main__":
    benchmark_outofcore()
