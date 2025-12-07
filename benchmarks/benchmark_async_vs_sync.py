import time
import numpy as np
import pycauset

def run_benchmark():
    if not pycauset.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    # Matrix Size
    # We want a size that forces tiling but fits in RAM.
    # 4000x4000 doubles = 128MB.
    # 3 matrices = 384MB.
    # If we set limit to 100MB, it MUST tile.
    N = 4000
    limit = 100 * 1024 * 1024 # 100 MB
    
    print(f"Generating {N}x{N} matrices...")
    A_np = np.random.rand(N, N).astype(np.float64)
    B_np = np.random.rand(N, N).astype(np.float64)
    
    A = pycauset.asarray(A_np)
    B = pycauset.asarray(B_np)
    
    print("-" * 60)
    print(f"Benchmark: MatMul {N}x{N} (Float64)")
    print(f"Memory Limit: {limit / 1024 / 1024:.2f} MB (Forces Tiling)")
    print("-" * 60)

    # 1. Synchronous Mode (Simulated Old Behavior)
    print("Running SYNCHRONOUS mode (enable_async=False)...")
    pycauset.cuda.enable(memory_limit=limit, enable_async=False)
    
    # Warmup
    _ = A @ B
    
    start_sync = time.time()
    C_sync = A @ B
    time_sync = time.time() - start_sync
    print(f"Sync Time: {time_sync:.4f} s")
    
    # 2. Asynchronous Mode (New Behavior)
    print("Running ASYNCHRONOUS mode (enable_async=True)...")
    pycauset.cuda.enable(memory_limit=limit, enable_async=True)
    
    # Warmup
    _ = A @ B
    
    start_async = time.time()
    C_async = A @ B
    time_async = time.time() - start_async
    print(f"Async Time: {time_async:.4f} s")
    
    # Results
    print("-" * 60)
    speedup = time_sync / time_async
    print(f"Speedup: {speedup:.2f}x")
    print("-" * 60)
    
    if speedup > 1.05:
        print("SUCCESS: Async Pipelining is providing a speedup.")
    else:
        print("WARNING: No significant speedup detected. Check PCIe bandwidth or matrix size.")

if __name__ == "__main__":
    run_benchmark()
