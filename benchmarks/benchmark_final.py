import time
import os
import sys

# Add python package to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

import pycauset

# Add package directory to DLL search path for Windows
if os.name == 'nt':
    package_dir = os.path.dirname(pycauset.__file__)
    os.add_dll_directory(package_dir)
    os.environ['PATH'] = package_dir + os.pathsep + os.environ['PATH']

def run_benchmark():
    print("="*60)
    print(f"PyCauset Benchmark - v{pycauset.__version__}")
    print("="*60)

    # 1. Check GPU
    print("\n[1] Checking Hardware Acceleration...")
    try:
        # Force GPU init
        pycauset.cuda.enable()
        if pycauset.cuda.is_available():
            print("SUCCESS: GPU backend initialized.")
        else:
            print("WARNING: GPU initialization failed (or not supported).")
            print("Falling back to CPU.")
    except Exception as e:
        print(f"WARNING: GPU initialization threw exception: {e}")
        print("Falling back to CPU.")

    device = pycauset.cuda.current_device()
    print(f"Active Device: {device}")

    # 2. Benchmark
    N = 4000
    print(f"\n[2] Running Matrix Multiplication Benchmark (N={N})...")
    
    print("Generating random matrices (CausalMatrix)...")
    t0 = time.time()
    # Use CausalMatrix (TriangularBitMatrix)
    A = pycauset.CausalMatrix.random(N, p=0.1)
    B = pycauset.CausalMatrix.random(N, p=0.1)
    print(f"Generation took: {time.time() - t0:.4f}s")

    print("Multiplying (MatMul)...")
    t0 = time.time()
    C = A @ B
    dt = time.time() - t0
    print(f"Multiplication took: {dt:.4f}s")
    
    ops = N**3  # Rough approximation for dense, bit matrix is different but good for relative
    print(f"Performance: {ops / dt / 1e9:.2f} GOps/s (Effective)")

    print("\n" + "="*60)
    if "cpu" in device.lower():
        print("NOTE: Benchmark ran on CPU. GPU was not available.")
        print("      If you have a GPU, check compatibility (Pascal/GTX 10 series not supported by CUDA 13).")
    else:
        print("Benchmark ran on GPU.")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()
