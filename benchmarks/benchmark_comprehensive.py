import time
import os
import sys
import platform
import numpy as np

# Add package directory to DLL search path for Windows
try:
    import pycauset
    if os.name == 'nt':
        package_dir = os.path.dirname(pycauset.__file__)
        os.add_dll_directory(package_dir)
        os.environ['PATH'] = package_dir + os.pathsep + os.environ['PATH']
except ImportError:
    # If running from source root without install
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))
    import pycauset
    if os.name == 'nt':
        package_dir = os.path.dirname(pycauset.__file__)
        try:
            os.add_dll_directory(package_dir)
            os.environ['PATH'] = package_dir + os.pathsep + os.environ['PATH']
        except Exception:
            pass

def setup_mode(mode):
    if mode == "cpu_seq":
        pycauset.set_num_threads(1)
        pycauset.cuda.disable()
        return True
    elif mode == "cpu_par":
        pycauset.set_num_threads(os.cpu_count())
        pycauset.cuda.disable()
        return True
    elif mode == "gpu":
        try:
            pycauset.cuda.enable()
            if not pycauset.cuda.is_available():
                print(" [SKIPPED: GPU Not Available]")
                return False
            return True
        except Exception as e:
            print(f" [SKIPPED: GPU Error {e}]")
            return False
    return False

def run_matmul_test(N, mode, label):
    print(f"  [MatMul] {label} (N={N})...", end="", flush=True)
    
    if not setup_mode(mode):
        return None

    # Generate Data (CausalMatrix / TriangularBitMatrix)
    t_gen = time.time()
    A = pycauset.CausalMatrix.random(N, p=0.1)
    B = pycauset.CausalMatrix.random(N, p=0.1)
    dt_gen = time.time() - t_gen
    
    # Warmup
    try:
        t_warm = time.time()
        _ = A @ B
        dt_warm = time.time() - t_warm
    except Exception as e:
        print(f" [FAILED: {e}]")
        return None
    
    # Benchmark
    t0 = time.time()
    C = A @ B
    dt = time.time() - t0
    
    ops = (N**3) / 1e9 
    gops = ops / dt
    
    print(f" Done. (Gen: {dt_gen:.2f}s, Warmup: {dt_warm:.2f}s, Run: {dt:.4f}s) -> {gops:.2f} GOps/s")
    return dt

def run_inverse_test(N, mode, label, dtype="float64"):
    print(f"  [Inverse {dtype}] {label} (N={N})...", end="", flush=True)
    
    if not setup_mode(mode):
        return None

    # Generate Data (DenseMatrix)
    t_gen = time.time()
    if dtype == "float64":
        if hasattr(pycauset, "Float64Matrix"):
            A = pycauset.Float64Matrix.random(N)
        else:
            A = pycauset.FloatMatrix.random(N)
    elif dtype == "float32":
        if hasattr(pycauset, "Float32Matrix"):
            A = pycauset.Float32Matrix.random(N)
        else:
            A = pycauset.FloatMatrix.random(N)
    dt_gen = time.time() - t_gen
            
    # Warmup
    try:
        t_warm = time.time()
        if hasattr(A, "inverse"):
            _ = A.inverse()
        elif hasattr(A, "invert"):
            _ = A.invert()
        else:
            print(" [FAILED: No inverse method]")
            return None
        dt_warm = time.time() - t_warm
    except Exception as e:
        print(f" [FAILED: {e}]")
        return None
    
    # Benchmark
    t0 = time.time()
    if hasattr(A, "inverse"):
        C = A.inverse()
    else:
        C = A.invert()
    dt = time.time() - t0
    
    ops = (N**3) / 1e9 # Approx for LU+Inverse
    gops = ops / dt
    
    print(f" Done. (Gen: {dt_gen:.2f}s, Warmup: {dt_warm:.2f}s, Run: {dt:.4f}s) -> {gops:.2f} GOps/s")
    return dt

def main():
    print("="*80)
    print(f"PyCauset Comprehensive Benchmark")
    print(f"Version: {pycauset.__version__}")
    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CPU Cores: {os.cpu_count()}")
    try:
        print(f"GPU Status: {pycauset.cuda.current_device() if pycauset.cuda.is_available() else 'Inactive'}")
    except:
        print("GPU Status: Unknown")
    print("="*80)

    # Sizes for MatMul
    matmul_sizes = [2000, 4000, 8000, 10000]
    
    # Sizes for Inverse (usually slower, so maybe smaller max)
    inverse_sizes = [2000, 4000, 8000]

    print("\n--- Matrix Multiplication (TriangularBitMatrix) ---")
    for N in matmul_sizes:
        print(f"\nSize: {N}x{N}")
        t_seq = run_matmul_test(N, "cpu_seq", "CPU Seq")
        t_par = run_matmul_test(N, "cpu_par", "CPU Par")
        t_gpu = run_matmul_test(N, "gpu",     "GPU    ")
        
        if t_seq and t_par: print(f"  -> CPU Speedup: {t_seq/t_par:.2f}x")
        if t_par and t_gpu: print(f"  -> GPU Speedup: {t_par/t_gpu:.2f}x")

    print("\n--- Matrix Inversion (Float64) ---")
    for N in inverse_sizes:
        print(f"\nSize: {N}x{N}")
        t_seq = run_inverse_test(N, "cpu_seq", "CPU Seq", "float64")
        t_par = run_inverse_test(N, "cpu_par", "CPU Par", "float64")
        t_gpu = run_inverse_test(N, "gpu",     "GPU    ", "float64")
        
        if t_seq and t_par: print(f"  -> CPU Speedup: {t_seq/t_par:.2f}x")
        if t_par and t_gpu: print(f"  -> GPU Speedup: {t_par/t_gpu:.2f}x")

    print("\n--- Matrix Inversion (Float32) ---")
    for N in inverse_sizes:
        print(f"\nSize: {N}x{N}")
        t_seq = run_inverse_test(N, "cpu_seq", "CPU Seq", "float32")
        t_par = run_inverse_test(N, "cpu_par", "CPU Par", "float32")
        t_gpu = run_inverse_test(N, "gpu",     "GPU    ", "float32")
        
        if t_seq and t_par: print(f"  -> CPU Speedup: {t_seq/t_par:.2f}x")
        if t_par and t_gpu: print(f"  -> GPU Speedup: {t_par/t_gpu:.2f}x")

    print("\n" + "="*80)
    print("Benchmark Complete.")
    print("="*80)

if __name__ == "__main__":
    main()
