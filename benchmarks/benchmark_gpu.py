import pycauset
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

def benchmark_matrix_mult():
    sizes = [512, 1024, 2048, 4096]
    results = []

    print(f"{'Size':<10} | {'Type':<15} | {'Time (s)':<10} | {'GFLOPS':<10}")
    print("-" * 55)

    for N in sizes:
        # 1. Float64 (Double)
        A64 = pycauset.FloatMatrix.random(N)
        B64 = pycauset.FloatMatrix.random(N)
        
        start = time.time()
        C64 = A64 @ B64
        end = time.time()
        t64 = end - start
        gflops64 = (2 * N**3) / (t64 * 1e9)
        
        results.append({"Size": N, "Type": "Float64", "Time": t64, "GFLOPS": gflops64})
        print(f"{N:<10} | {'Float64':<15} | {t64:<10.4f} | {gflops64:<10.2f}")

        # 2. Float32 (Single)
        A32 = pycauset.Float32Matrix.random(N)
        B32 = pycauset.Float32Matrix.random(N)
        
        start = time.time()
        C32 = A32 @ B32
        end = time.time()
        t32 = end - start
        gflops32 = (2 * N**3) / (t32 * 1e9)
        
        results.append({"Size": N, "Type": "Float32", "Time": t32, "GFLOPS": gflops32})
        print(f"{N:<10} | {'Float32':<15} | {t32:<10.4f} | {gflops32:<10.2f}")

        # 3. DenseBitMatrix (Boolean)
        # Note: GFLOPS isn't the right metric, maybe GOPs (Gigabit Operations)
        # 64 ops per cycle effectively.
        ABit = pycauset.DenseBitMatrix.random(N)
        BBit = pycauset.DenseBitMatrix.random(N)
        
        start = time.time()
        CBit = ABit @ BBit
        end = time.time()
        tBit = end - start
        # Ops = N^3 (logical AND/ORs)
        gops = (N**3) / (tBit * 1e9)
        
        results.append({"Size": N, "Type": "BitMatrix", "Time": tBit, "GFLOPS": gops})
        print(f"{N:<10} | {'BitMatrix':<15} | {tBit:<10.4f} | {gops:<10.2f} (GOPS)")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    if not pycauset.is_gpu_available():
        print("WARNING: GPU not detected. Benchmarks will run on CPU (slow).")
    
    benchmark_matrix_mult()
