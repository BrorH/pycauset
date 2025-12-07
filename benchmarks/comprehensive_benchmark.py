import time
import pycauset
import numpy as np
import os
import sys

# Configuration
SIZES = [500, 1000, 2000]
MODES = ["Sequential", "CPU Parallel", "GPU"]
OPERATIONS = ["MatMul", "Inverse", "Eigvals", "Arnoldi"]

def setup_matrix(N, name):
    """Create a random FloatMatrix."""
    mat = pycauset.FloatMatrix(N, f"bench_{name}.bin")
    # Use numpy to generate data quickly
    data = np.random.rand(N, N)
    # For Eigvals, make it symmetric
    if name == "sym":
        data = data + data.T
    
    # Fill matrix (this might be slow for large N, but it's setup time)
    # We can optimize by writing directly to file if we knew the format, 
    # but let's use the API.
    for i in range(N):
        for j in range(N):
            mat.set(i, j, data[i, j])
    return mat

def run_benchmark():
    results = []
    
    print(f"{'Operation':<15} | {'Size':<6} | {'Mode':<15} | {'Time (s)':<10}")
    print("-" * 55)

    for N in SIZES:
        # Setup matrices once per size to avoid I/O overhead in measurement
        # MatMul
        A = setup_matrix(N, "A")
        B = setup_matrix(N, "B")
        # Symmetric for Eigvals
        S = setup_matrix(N, "sym")
        
        for mode in MODES:
            # Configure Mode
            if mode == "Sequential":
                pycauset.cuda.disable()
                pycauset.set_num_threads(1)
            elif mode == "CPU Parallel":
                pycauset.cuda.disable()
                # Use all available cores
                count = os.cpu_count() or 4
                pycauset.set_num_threads(count)
            elif mode == "GPU":
                pycauset.cuda.enable()
                if not pycauset.cuda.is_available():
                    results.append((N, mode, "MatMul", "N/A"))
                    results.append((N, mode, "Inverse", "N/A"))
                    results.append((N, mode, "Eigvals", "N/A"))
                    results.append((N, mode, "Arnoldi", "N/A"))
                    continue

            # MatMul
            start = time.time()
            C = A.multiply(B, "bench_C.bin")
            duration = time.time() - start
            C.close()
            results.append((N, mode, "MatMul", duration))
            print(f"{'MatMul':<15} | {N:<6} | {mode:<15} | {duration:.4f}")

            # Inverse
            # Only run for smaller sizes if it's too slow? 
            # Inverse is O(N^3), same as MatMul.
            start = time.time()
            try:
                # Use A (random) - likely invertible
                # Add identity to ensure invertibility
                # We can't easily modify A in place efficiently via API loop
                # Just try inverting A.
                A_inv = A.invert()
                duration = time.time() - start
                A_inv.close()
            except Exception as e:
                duration = -1.0 # Error
            results.append((N, mode, "Inverse", duration))
            print(f"{'Inverse':<15} | {N:<6} | {mode:<15} | {duration:.4f}")

            # Eigvals (Symmetric)
            start = time.time()
            ev = pycauset.eigvals(S)
            duration = time.time() - start
            results.append((N, mode, "Eigvals", duration))
            print(f"{'Eigvals':<15} | {N:<6} | {mode:<15} | {duration:.4f}")

            # Arnoldi (Top k=10)
            start = time.time()
            ev_arn = pycauset.eigvals_arnoldi(A, 10)
            duration = time.time() - start
            results.append((N, mode, "Arnoldi", duration))
            print(f"{'Arnoldi':<15} | {N:<6} | {mode:<15} | {duration:.4f}")

        A.close()
        B.close()
        S.close()
        
        # Cleanup files
        for f in os.listdir("."):
            if f.startswith("bench_") and f.endswith(".bin"):
                try:
                    os.remove(f)
                except:
                    pass

    return results

def generate_report(results):
    with open("BENCHMARK_REPORT.md", "w") as f:
        f.write("# PyCauset Performance Benchmark Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n")
        f.write(f"**System:** {sys.platform}\n")
        f.write(f"**CPU Cores:** {os.cpu_count()}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Operation | Size | Mode | Time (s) |\n")
        f.write("|---|---|---|---|\n")
        
        for N, mode, op, duration in results:
            dur_str = f"{duration:.4f}" if isinstance(duration, float) and duration > 0 else str(duration)
            f.write(f"| {op} | {N} | {mode} | {dur_str} |\n")

        f.write("\n## Analysis\n\n")
        f.write("(Auto-generated analysis placeholder)\n")

if __name__ == "__main__":
    print("Starting Benchmark...")
    results = run_benchmark()
    generate_report(results)
    print("\nBenchmark Complete. Report saved to BENCHMARK_REPORT.md")
