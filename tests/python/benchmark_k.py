import time
import numpy as np
from pycauset import CausalMatrix, compute_k
import os

def benchmark(N, a=1.0):
    print(f"Benchmarking N={N}...")
    
    # Create random matrix
    start = time.time()
    C = CausalMatrix.random(N, 0.1, backing_file=f"bench_{N}.pycauset")
    print(f"  Creation: {time.time() - start:.4f}s")
    
    # C++ Implementation
    start = time.time()
    K = compute_k(C, a)
    cpp_time = time.time() - start
    print(f"  C++ Time: {cpp_time:.4f}s")
    
    K.close()
    
    # NumPy Implementation (only for small N)
    if N <= 2000:
        start = time.time()
        C_np = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if C.get(i, j):
                    C_np[i, j] = 1.0
        
        I = np.eye(N)
        K_ref = C_np @ np.linalg.inv(a * I + C_np)
        numpy_time = time.time() - start
        print(f"  NumPy Time: {numpy_time:.4f}s")
        print(f"  Speedup: {numpy_time / cpp_time:.2f}x")
    
    C.close()

if __name__ == "__main__":
    benchmark(500)
    benchmark(1000)
