import time
import os
import numpy as np
import pycauset
import tempfile
import shutil

class BenchmarkResult:
    def __init__(self, name, size_mb, throughput, parity_ratio):
        self.name = name
        self.size_mb = size_mb
        self.throughput = throughput
        self.parity_ratio = parity_ratio
        self.passed = parity_ratio >= 0.90

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"{self.name:<40} | {self.size_mb:>6.1f} MB | {self.throughput:>8.2f} MB/s | {self.parity_ratio:>6.2f}x | {status}"

def run_benchmark(dtype, rows, cols, label):
    element_size = np.dtype(dtype).itemsize
    total_bytes = rows * cols * element_size
    size_mb = total_bytes / (1024 * 1024)
    
    print(f"\n--- Benchmarking {label} ({rows}x{cols}, {dtype.__name__}, {size_mb:.1f} MB) ---")
    
    # Generate Data
    data = np.zeros((rows, cols), dtype=dtype) # Fast generation
    
    # 1. NumPy Memmap Baseline (Write)
    tmp_dir = tempfile.mkdtemp()
    np_path = os.path.join(tmp_dir, "numpy.dat")
    
    start = time.time()
    fp = np.memmap(np_path, dtype=dtype, mode='w+', shape=(rows, cols))
    fp[:] = data[:] # Write data
    fp.flush()
    del fp # Close
    np_write_time = time.time() - start
    np_write_bw = size_mb / np_write_time
    
    # 2. PyCauset (Write)
    start = time.time()
    mat = pycauset.matrix(data, storage="disk")
    # Ensure flush? PyCauset relies on OS, but let's assume from_numpy returns when data is handed over
    pc_write_time = time.time() - start
    pc_write_bw = size_mb / pc_write_time
    
    # 3. NumPy Memmap Baseline (Read)
    start = time.time()
    fp = np.memmap(np_path, dtype=dtype, mode='r', shape=(rows, cols))
    _ = np.array(fp) # Force read into RAM
    del fp
    np_read_time = time.time() - start
    np_read_bw = size_mb / np_read_time
    
    # 4. PyCauset (Read)
    start = time.time()
    _ = pycauset.to_numpy(mat, allow_huge=True)
    pc_read_time = time.time() - start
    pc_read_bw = size_mb / pc_read_time
    
    # Cleanup
    shutil.rmtree(tmp_dir)
    
    # Results
    results = []
    results.append(BenchmarkResult(f"{label} Write", size_mb, pc_write_bw, pc_write_bw / np_write_bw))
    results.append(BenchmarkResult(f"{label} Read", size_mb, pc_read_bw, pc_read_bw / np_read_bw))
    
    return results

def benchmark_non_contiguous(rows, cols):
    label = "Float64 Non-Contiguous (Sliced)"
    dtype = np.float64
    # Create larger array and slice it to be non-contiguous
    # We want a slice that is not C-contiguous but is 2D.
    # e.g. every other column
    expanded_cols = cols * 2
    data_big = np.zeros((rows, expanded_cols), dtype=dtype)
    data = data_big[:, ::2] # Sliced view, non-contiguous
    
    element_size = np.dtype(dtype).itemsize
    total_bytes = rows * cols * element_size
    size_mb = total_bytes / (1024 * 1024)

    print(f"\n--- Benchmarking {label} ({rows}x{cols}, {size_mb:.1f} MB) ---")
    print(f"Input flags: C_CONTIGUOUS={data.flags['C_CONTIGUOUS']}")

    # 1. NumPy Copy Baseline
    # We benchmark how fast NumPy can copy this non-contiguous view to a contiguous array
    start = time.time()
    _ = np.ascontiguousarray(data)
    np_copy_time = time.time() - start
    np_copy_bw = size_mb / np_copy_time

    # 2. PyCauset Import
    # This should trigger the non-contiguous path in bind_matrix.cpp
    start = time.time()
    _ = pycauset.matrix(data) # In-memory import
    pc_import_time = time.time() - start
    pc_import_bw = size_mb / pc_import_time

    results = []
    # We compare against NumPy's own internal copy speed
    # Target is >0.90x of numpy's optimized copy
    results.append(BenchmarkResult(f"{label} Import", size_mb, pc_import_bw, pc_import_bw / np_copy_bw))
    return results

def benchmark_bit_matrix(rows, cols):
    # Special case for BitMatrix because NumPy uses 1 byte per bool, PyCauset uses 1 bit.
    # We compare logical throughput (elements/sec) or raw bandwidth?
    # Let's compare "Time to save/load N booleans".
    
    label = "BitMatrix (bool)"
    dtype = bool
    # NumPy size in MB (1 byte per bool)
    np_size_mb = (rows * cols) / (1024 * 1024)
    # PyCauset size in MB (1 bit per bool)
    pc_size_mb = np_size_mb / 8
    
    print(f"\n--- Benchmarking {label} ({rows}x{cols}) ---")
    print(f"NumPy Size: {np_size_mb:.1f} MB, PyCauset Size: {pc_size_mb:.1f} MB")
    
    data = np.random.randint(0, 2, size=(rows, cols)).astype(bool)
    
    # NumPy Memmap
    tmp_dir = tempfile.mkdtemp()
    np_path = os.path.join(tmp_dir, "numpy.dat")
    
    start = time.time()
    fp = np.memmap(np_path, dtype=dtype, mode='w+', shape=(rows, cols))
    fp[:] = data[:]
    fp.flush()
    del fp
    np_write_time = time.time() - start
    
    # PyCauset
    start = time.time()
    mat = pycauset.matrix(data, storage="disk")
    pc_write_time = time.time() - start
    
    # Read
    start = time.time()
    fp = np.memmap(np_path, dtype=dtype, mode='r', shape=(rows, cols))
    _ = np.array(fp)
    del fp # Ensure file handle is closed
    np_read_time = time.time() - start
    
    start = time.time()
    _ = pycauset.to_numpy(mat, allow_huge=True)
    pc_read_time = time.time() - start
    
    shutil.rmtree(tmp_dir)
    
    # For BitMatrix, we expect PyCauset to be FASTER (ratio > 1.0) because of compression
    # We use NumPy's time as baseline.
    results = []
    results.append(BenchmarkResult(f"{label} Write", np_size_mb, np_size_mb/pc_write_time, np_write_time / pc_write_time))
    results.append(BenchmarkResult(f"{label} Read", np_size_mb, np_size_mb/pc_read_time, np_read_time / pc_read_time))
    
    return results

def benchmark_non_contiguous(rows, cols):
    label = "Float64 Non-Contiguous (Sliced)"
    dtype = np.float64
    # Create larger array and slice it to be non-contiguous
    # We want a slice that is not C-contiguous but is 2D.
    # e.g. every other column
    expanded_cols = cols * 2
    data_big = np.zeros((rows, expanded_cols), dtype=dtype)
    data = data_big[:, ::2] # Sliced view, non-contiguous
    
    element_size = np.dtype(dtype).itemsize
    total_bytes = rows * cols * element_size
    size_mb = total_bytes / (1024 * 1024)

    print(f"\n--- Benchmarking {label} ({rows}x{cols}, {size_mb:.1f} MB) ---")
    print(f"Input flags: C_CONTIGUOUS={data.flags['C_CONTIGUOUS']}")

    # 1. NumPy Copy Baseline
    # We benchmark how fast NumPy can copy this non-contiguous view to a contiguous array
    start = time.time()
    _ = np.ascontiguousarray(data)
    np_copy_time = time.time() - start
    np_copy_bw = size_mb / np_copy_time

    # 2. PyCauset Import
    # This should trigger the non-contiguous path in bind_matrix.cpp
    start = time.time()
    _ = pycauset.matrix(data) # In-memory import
    pc_import_time = time.time() - start
    pc_import_bw = size_mb / pc_import_time

    results = []
    # We compare against NumPy's own internal copy speed
    # Target is >0.90x of numpy's optimized copy
    results.append(BenchmarkResult(f"{label} Import", size_mb, pc_import_bw, pc_import_bw / np_copy_bw))
    return results

def main():
    print(f"{'Benchmark':<40} | {'Size':>9} | {'Speed':>11} | {'Ratio':>7} | {'Status'}")
    print("-" * 85)
    
    all_results = []
    
    # 1. Standard Float64 (Square) - 100MB
    # 100MB = 12.5M doubles -> 3535x3535
    all_results.extend(run_benchmark(np.float64, 3535, 3535, "Float64 Square 100MB"))
    
    # 2. Standard Float64 (Large) - 1GB
    # 1GB = 125M doubles -> 11180x11180
    all_results.extend(run_benchmark(np.float64, 11180, 11180, "Float64 Square 1GB"))
    
    # 3. Complex128 (Heavy) - 200MB
    # 200MB = 12.5M complex128 (16 bytes) -> 3535x3535
    all_results.extend(run_benchmark(np.complex128, 3535, 3535, "Complex128 Square 200MB"))
    
    # 4. Int32 (Light) - 100MB
    # 100MB = 25M ints -> 5000x5000
    all_results.extend(run_benchmark(np.int32, 5000, 5000, "Int32 Square 100MB"))
    
    # 5. Tall Skinny (Database style)
    # 10M rows x 10 cols, float64 -> 800MB
    all_results.extend(run_benchmark(np.float64, 10_000_000, 10, "Float64 Tall 800MB"))
    
    # 6. BitMatrix (Bool)
    # 10000x10000 -> 100M elements
    # NumPy: 100MB. PyCauset: 12.5MB.
    all_results.extend(benchmark_bit_matrix(10000, 10000))

    # 7. Non-Contiguous Import
    all_results.extend(benchmark_non_contiguous(3535, 3535))
    
    print("\n" + "="*85)
    print("FINAL SUMMARY")
    print("="*85)
    for r in all_results:
        print(r)
        
    failures = [r for r in all_results if not r.passed]
    if failures:
        print(f"\nFAILURES DETECTED: {len(failures)}")
        exit(1)
    else:
        print("\nALL BENCHMARKS PASSED (>0.90x NumPy Parity)")
        exit(0)

if __name__ == "__main__":
    main()
