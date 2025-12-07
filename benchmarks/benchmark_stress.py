import time
import numpy as np
import pycauset
import sys
import os

def stress_test():
    if not pycauset.cuda.is_available():
        print("CUDA not available.")
        return

    print("\n" + "="*80)
    print("GPU OUT-OF-CORE STRESS TEST")
    print("="*80)
    print("Objective: Push the system to maximum throughput with limited VRAM.")
    print("Target: > 1.5 TFLOPS (Sustained)")
    print("-" * 80)

    # Setup: Large Matrix
    # N=14000 requires ~2.2GB. 
    N = 14000
    limit = 200 * 1024 * 1024 # 200 MB Limit
    
    print(f"Matrix Size: {N}x{N}")
    print(f"Total Memory Required: {3 * N**2 * 4 / 1024**3:.2f} GB")
    print(f"VRAM Limit Enforced: {limit/1024/1024:.0f} MB")
    print(f"Constraint: System MUST stream data. Entire matrix cannot fit.")
    
    print("\nGenerating Data...")
    A_np = np.random.rand(N, N).astype(np.float32)
    B_np = np.random.rand(N, N).astype(np.float32)
    A = pycauset.asarray(A_np)
    B = pycauset.asarray(B_np)
    
    # Test 1: Synchronous (Baseline)
    print("\n[Test 1: Synchronous Execution]")
    pycauset.cuda.enable(memory_limit=limit, enable_async=False)
    
    start = time.time()
    C_sync = A @ B
    time_sync = time.time() - start
    print(f"Time: {time_sync:.4f} s")
    
    # Test 2: Asynchronous (Optimized)
    print("\n[Test 2: Asynchronous Execution (Hybrid CPU/GPU)]")
    pycauset.cuda.enable(memory_limit=limit, enable_async=True)
    
    start = time.time()
    C_async = A @ B
    time_async = time.time() - start
    print(f"Time: {time_async:.4f} s")
    
    # Analysis
    print("-" * 60)
    flops = 2 * N**3
    tflops_sync = (flops / time_sync) / 1e12
    tflops_async = (flops / time_async) / 1e12
    
    print(f"Throughput (Sync):  {tflops_sync:.2f} TFLOPS")
    print(f"Throughput (Async): {tflops_async:.2f} TFLOPS")
    
    speedup = time_sync / time_async
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Estimate Peak based on typical GTX 1060 (4.4 TFLOPS)
    peak_tflops = 4.4 
    efficiency = tflops_async / peak_tflops * 100
    print(f"Efficiency: {efficiency:.1f}% of Theoretical Peak (4.4 TFLOPS)")
    
    if tflops_async > 1.5:
        print("\nCONCLUSION: PASSED")
        print("The system is sustaining high throughput despite the VRAM bottleneck.")
        print("This confirms the streaming architecture is efficient.")
    else:
        print("\nCONCLUSION: UNDERPERFORMING")

if __name__ == "__main__":
    stress_test()
