import time
import numpy as np
import os
from pycauset import Float16Matrix, Float32Matrix, FloatMatrix, eigvals_arnoldi

def benchmark_matrix_type(matrix_class, n, k=10, max_iter=30):
    bytes_per_elem = 8
    if "Float32" in matrix_class.__name__: bytes_per_elem = 4
    if "Float16" in matrix_class.__name__: bytes_per_elem = 2
    size_gb = (n * n * bytes_per_elem) / (1024**3)
    print(f"Benchmarking {matrix_class.__name__} with N={n} ({size_gb:.2f} GB)...")
    
    filename = f"bench_{matrix_class.__name__}_{n}"
    # Cleanup previous run
    if os.path.exists(filename + ".matrix"):
        try:
            os.remove(filename + ".matrix")
        except:
            pass
            
    start_setup = time.time()
    m = matrix_class(n, filename)
    
    # Fill with some data (random-ish but deterministic for fair comparison)
    # We can't easily fill efficiently from python loop for large N, 
    # so we'll just set diagonal and some off-diagonal to make it interesting
    # but sparse-ish updates to save setup time.
    # Actually, let's just set the diagonal. Arnoldi doesn't care much about values for speed, just operations.
    # But to avoid breakdown, we need some non-zero vectors.
    # Let's set diagonal.
    for i in range(n):
        m.set(i, i, float(i))
        if i < n-1:
            m.set(i, i+1, 0.5)
            m.set(i+1, i, 0.5)
            
    setup_time = time.time() - start_setup
    print(f"  Setup time: {setup_time:.4f}s")
    
    start_solve = time.time()
    # Use a larger tolerance to ensure it converges or runs enough iterations
    eigvals_arnoldi(m, k=k, max_iter=max_iter, tol=1e-5)
    solve_time = time.time() - start_solve
    print(f"  Solve time: {solve_time:.4f}s")
    
    m.close()
    # Cleanup
    if os.path.exists(filename + ".matrix"):
        try:
            os.remove(filename + ".matrix")
        except:
            pass
            
    return setup_time, solve_time

def run_benchmark():
    # N = 40000 to test larger scale performance (approx 12GB for Double)
    N = 40000 
    K = 20
    MAX_ITER = 60 # Ensure we do some work
    
    print(f"Starting Benchmark (N={N}, k={K}, max_iter={MAX_ITER})")
    print("-" * 50)
    
    t_f64 = benchmark_matrix_type(FloatMatrix, N, K, MAX_ITER)
    t_f32 = benchmark_matrix_type(Float32Matrix, N, K, MAX_ITER)
    t_f16 = benchmark_matrix_type(Float16Matrix, N, K, MAX_ITER)
    
    print("-" * 50)
    print("Results Summary (Solve Time):")
    print(f"Float64 (Double): {t_f64[1]:.4f}s")
    print(f"Float32 (Single): {t_f32[1]:.4f}s  (Speedup vs F64: {t_f64[1]/t_f32[1]:.2f}x)")
    print(f"Float16 (Half):   {t_f16[1]:.4f}s  (Speedup vs F64: {t_f64[1]/t_f16[1]:.2f}x)")

if __name__ == "__main__":
    run_benchmark()
