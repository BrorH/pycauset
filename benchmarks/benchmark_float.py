import time
import sys
import os
import numpy as np

# Force use of local python directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

import pycauset
print(f"Using pycauset from: {pycauset.__file__}")

if os.name == 'nt':
    package_dir = os.path.dirname(pycauset.__file__)
    try:
        os.add_dll_directory(package_dir)
    except AttributeError:
        pass
    os.environ['PATH'] = package_dir + os.pathsep + os.environ['PATH']

# Ensure pycauset.cuda is available
try:
    import pycauset.cuda
except ImportError:
    pass 

def benchmark(n):
    print(f"Benchmarking N={n}")
    
    # Create random matrices
    print("Creating matrices...")
    # Use float64 for FloatMatrix (DenseMatrix<double>)
    a_np = np.random.rand(n, n).astype(np.float64)
    b_np = np.random.rand(n, n).astype(np.float64)
    
    a = pycauset.asarray(a_np)
    b = pycauset.asarray(b_np)
    
    # CPU Benchmark
    pycauset.cuda.disable()
    print("Running CPU benchmark...")
    start = time.time()
    c_cpu = a.multiply(b)
    end = time.time()
    cpu_time = end - start
    print(f"CPU Time: {cpu_time:.4f} seconds")
    
    # GPU Benchmark
    # Check if GPU is available (bindings showed is_available in cuda submodule)
    pycauset.cuda.enable()
    if pycauset.cuda.is_available():
        print(f"Running GPU benchmark on {pycauset.cuda.current_device()}...")
        # Warmup
        a.multiply(b)
        
        start = time.time()
        c_gpu = a.multiply(b)
        end = time.time()
        gpu_time = end - start
        print(f"GPU Time: {gpu_time:.4f} seconds")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("GPU not available.")
    
    print("-" * 20)

if __name__ == "__main__":
    benchmark(2000)
    benchmark(4000)
    benchmark(6000)
