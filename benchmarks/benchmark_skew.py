import pycauset
import numpy as np
import time
import os

def benchmark():
    print("Benchmarking Skew-Symmetric Eigenvalue Solver...")
    
    # Parameters
    N = 30000  # Even larger matrix for stress testing
    K = 20
    MAX_ITER = 100
    
    print(f"Generating random skew-symmetric matrix (N={N})...")
    # Create a temporary file for the matrix
    mat_path = f"benchmark_skew_{N}.mat"
    if os.path.exists(mat_path):
        os.remove(mat_path)
        
    # We use pycauset to create it directly to avoid huge numpy array in memory if possible,
    # but we need to fill it.
    # Let's create a DenseMatrix and fill it.
    # Actually, filling element by element from Python is slow.
    # Better to create a numpy array and convert? 
    # 30000^2 doubles is ~7.2 GB. Python can handle it if >16GB RAM.
    
    # Generate random matrix A
    # A = B - B.T
    # To save memory, we can generate chunks?
    # Or just rely on the machine having > 8GB RAM.
    
    try:
        # Create pycauset matrix directly
        # We can't easily fill it efficiently from Python without numpy.
        
        print(f"Allocating {N}x{N} matrix...")
        # Use float32 to save memory if needed, but solver uses doubles.
        # Let's stick to float64 (default rand)
        A_np = np.random.rand(N, N)
        A_np = A_np - A_np.T # Skew symmetric
        
        print("Converting to PyCauset matrix...")
        # Trick: Create zero matrix and add numpy array to it.
        # This uses the internal from_numpy conversion and parallel addition.
        zeros = pycauset.FloatMatrix(N)
        A = zeros + A_np
        
        # Free numpy memory
        del A_np
        
        # 1. Sequential Run
        print("\n--- Sequential Run (1 Thread) ---")
        pycauset.set_num_threads(1)
        start_time = time.time()
        vals_seq = pycauset.eigvals_skew(A, k=K, max_iter=MAX_ITER)
        end_time = time.time()
        seq_time = end_time - start_time
        print(f"Time: {seq_time:.4f} seconds")
        
        # 2. Parallel Run
        # Use all available cores
        num_threads = os.cpu_count() or 8
        print(f"\n--- Parallel Run ({num_threads} Threads) ---")
        pycauset.set_num_threads(num_threads)
        start_time = time.time()
        vals_par = pycauset.eigvals_skew(A, k=K, max_iter=MAX_ITER)
        end_time = time.time()
        par_time = end_time - start_time
        print(f"Time: {par_time:.4f} seconds")
        
        # Results
        speedup = seq_time / par_time
        print(f"\nSpeedup: {speedup:.2f}x")
        
        # Cleanup
        # A is a persistent object, we should delete its file if possible.
        # But pycauset manages it.
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

if __name__ == "__main__":
    benchmark()
