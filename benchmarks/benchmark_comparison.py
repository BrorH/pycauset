import pycauset
import time
import numpy as np
import sys

def benchmark_comparison():
    print("="*80)
    print("BENCHMARK: CPU vs GPU (Out-of-Core) Inverse")
    print("="*80)

    if not pycauset.cuda.is_available():
        # Try to enable it first to make sure we have a GPU to test
        try:
            pycauset.cuda.enable()
        except:
            pass
            
    if not pycauset.cuda.is_available():
        print("Error: CUDA is not available. Cannot run comparison benchmark.")
        return

    sizes = [512, 1024, 2048, 4096]
    
    print(f"{'Size':<10} | {'CPU Time (s)':<15} | {'GPU Time (s)':<15} | {'Speedup':<10}")
    print("-" * 60)

    for N in sizes:
        # Generate Data
        # We use numpy to generate and then copy to pycauset to ensure same data
        np.random.seed(42)
        # Use identity + random to ensure invertibility and stability
        A_np = np.eye(N) * 2.0 + np.random.rand(N, N) * 0.1
        
        # ---------------------------------------------------------
        # CPU Benchmark
        # ---------------------------------------------------------
        pycauset.cuda.disable()
        
        # Create matrix on CPU (storage is file-backed anyway, but processing is CPU)
        A_cpu = pycauset.Float64Matrix(N)
        # Copy data
        # Assuming we can set data via numpy or element-wise. 
        # For speed of setup, we'll just assume creation is fast enough or not count it.
        # Actually, let's count only the inverse time.
        
        # We need to populate A_cpu. 
        # Since there is no direct from_numpy for existing matrix, we might need to iterate or use a helper if available.
        # But wait, `pycauset.Float64Matrix(N)` creates a zero matrix.
        # Let's just use a diagonal matrix for speed of setup if possible, or random.
        # The bindings don't show a quick "from_numpy" constructor exposed directly as a constructor, 
        # but there is `from_numpy` helper in bindings.cpp but it returns unique_ptr.
        # Let's check if we can assign.
        
        # To avoid slow python loops, let's use the fact that we are benchmarking INVERSE.
        # The content matters for singularity, but for performance (dense), it shouldn't matter much 
        # unless the CPU solver has early exits (unlikely for standard LU).
        # However, to be safe, let's use a matrix that is definitely non-singular.
        # Identity is easiest to set if there's a method, but `Float64Matrix` is dense.
        
        # Let's use the `fill_diagonal` or similar if available, or just accept the python loop overhead for setup 
        # (outside the timer).
        
        for i in range(N):
            A_cpu[i, i] = 2.0
            if i < N-1:
                A_cpu[i, i+1] = 0.5 # Make it slightly non-trivial
        
        start_cpu = time.time()
        try:
            # This should run on CPU because we called disable()
            res_cpu = A_cpu.inverse()
            end_cpu = time.time()
            cpu_time = end_cpu - start_cpu
        except Exception as e:
            cpu_time = float('inf')
            print(f"CPU Failed: {e}")

        # ---------------------------------------------------------
        # GPU Benchmark
        # ---------------------------------------------------------
        pycauset.cuda.enable()
        
        # Create matrix again (or reuse, but better to start fresh to avoid caching artifacts if any)
        A_gpu = pycauset.Float64Matrix(N)
        for i in range(N):
            A_gpu[i, i] = 2.0
            if i < N-1:
                A_gpu[i, i+1] = 0.5
        
        # Warmup (optional, but good for GPU)
        if N == sizes[0]:
            try:
                A_gpu.inverse()
            except:
                pass
        
        start_gpu = time.time()
        try:
            res_gpu = A_gpu.inverse()
            end_gpu = time.time()
            gpu_time = end_gpu - start_gpu
        except Exception as e:
            gpu_time = float('inf')
            print(f"GPU Failed: {e}")

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
        
        print(f"{N:<10} | {cpu_time:<15.4f} | {gpu_time:<15.4f} | {speedup:<10.2f}x")

    print("\n" + "="*80)
    print("BENCHMARK: CPU vs GPU (Float32 / Single Precision)")
    print("Note: Consumer GPUs (GeForce) are much faster at Float32 than Float64.")
    print("="*80)
    print(f"{'Size':<10} | {'CPU Time (s)':<15} | {'GPU Time (s)':<15} | {'Speedup':<10}")
    print("-" * 60)

    for N in sizes:
        # CPU Benchmark (Float32)
        pycauset.cuda.disable()
        A_cpu = pycauset.Float32Matrix(N)
        # Fill diagonal (simple loop, overhead ignored as before)
        for i in range(N):
            A_cpu[i, i] = 2.0
            if i < N-1: A_cpu[i, i+1] = 0.5
            
        start_cpu = time.time()
        try:
            res_cpu = A_cpu.inverse()
            end_cpu = time.time()
            cpu_time = end_cpu - start_cpu
        except Exception as e:
            cpu_time = float('inf')
            # print(f"CPU Failed: {e}")

        # GPU Benchmark (Float32)
        pycauset.cuda.enable()
        A_gpu = pycauset.Float32Matrix(N)
        for i in range(N):
            A_gpu[i, i] = 2.0
            if i < N-1: A_gpu[i, i+1] = 0.5
            
        # Warmup
        if N == sizes[0]:
            try: A_gpu.inverse()
            except: pass

        start_gpu = time.time()
        try:
            res_gpu = A_gpu.inverse()
            end_gpu = time.time()
            gpu_time = end_gpu - start_gpu
        except Exception as e:
            gpu_time = float('inf')
            print(f"GPU Failed: {e}")

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
        print(f"{N:<10} | {cpu_time:<15.4f} | {gpu_time:<15.4f} | {speedup:<10.2f}x")

if __name__ == "__main__":
    benchmark_comparison()
