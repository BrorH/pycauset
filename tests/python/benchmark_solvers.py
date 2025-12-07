import time
import numpy as np
import pycauset
# import matplotlib.pyplot as plt

def benchmark_solvers():
    sizes = [50, 100, 200, 300, 400, 500, 1000]
    times_pc = []
    times_np = []
    
    print(f"{'N':<10} | {'PyCauset (s)':<15} | {'NumPy (s)':<15} | {'Speedup':<10}")
    print("-" * 60)
    
    for n in sizes:
        # Generate random complex matrix
        A_real = np.random.randn(n, n)
        A_imag = np.random.randn(n, n)
        A_np = A_real + 1j * A_imag
        
        # Create PyCauset matrix
        A_pc = pycauset.ComplexMatrix(n)
        for i in range(n):
            for j in range(n):
                A_pc.set(i, j, complex(A_np[i, j]))
        
        # Benchmark PyCauset
        start = time.time()
        pycauset.eig(A_pc)
        end = time.time()
        t_pc = end - start
        times_pc.append(t_pc)
        
        # Benchmark NumPy
        start = time.time()
        np.linalg.eig(A_np)
        end = time.time()
        t_np = end - start
        times_np.append(t_np)
        
        speedup = t_np / t_pc if t_pc > 0 else 0
        print(f"{n:<10} | {t_pc:<15.4f} | {t_np:<15.4f} | {speedup:<10.2f}x")

    print("\nBenchmark complete.")
    
    # Note: PyCauset currently copies data to Eigen (C++) and back.
    # NumPy uses optimized BLAS/LAPACK directly.
    # We expect PyCauset to be competitive but maybe slightly slower due to overhead,
    # or faster if Eigen's optimizations beat standard numpy build on this machine.

if __name__ == "__main__":
    benchmark_solvers()
