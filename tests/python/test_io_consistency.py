import pytest
import numpy as np
import pycauset
import os

# Parameterize over dtypes
@pytest.mark.parametrize("dtype", [
    np.float32, np.float64, 
    np.complex64, np.complex128, 
    np.int32
])
def test_io_roundtrip_consistency(dtype):
    """
    Verify that data written to disk and read back is identical to the source.
    This tests SetFileValidData/fallocate and mmap logic.
    """
    rows, cols = 1000, 1000
    
    # Create random data
    if np.issubdtype(dtype, np.complexfloating):
        data = np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)
        data = data.astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        data = np.random.randint(-1000, 1000, size=(rows, cols)).astype(dtype)
    else:
        data = np.random.rand(rows, cols).astype(dtype)
        
    # Write to PyCauset (disk-backed)
    # Use pycauset.matrix() factory
    mat = pycauset.matrix(data, storage="disk")
    
    # Read back
    result = pycauset.to_numpy(mat, allow_huge=True)
    
    # Verify
    np.testing.assert_array_equal(data, result, err_msg=f"Roundtrip failed for {dtype}")

def test_large_file_consistency():
    """
    Test a larger file (e.g. > 100MB) to ensure chunking/prefetching works.
    """
    # 100MB of float64 is ~12.5M elements. 3500x3500
    n = 3500
    dtype = np.float64
    
    # Use deterministic data to avoid memory pressure from random generation
    # np.full is efficient
    val = 3.14159
    data = np.full((n, n), val, dtype=dtype)
    
    mat = pycauset.matrix(data, storage="disk")
    
    # Check a few elements without full readback first (if API supports it)
    # Assuming we have to_numpy() for full read
    result = pycauset.to_numpy(mat, allow_huge=True)
    
    np.testing.assert_array_equal(data, result)

def test_direct_path_consistency():
    """
    Test that operations using the 'Direct Path' (RAM-resident) produce correct results.
    We force this by using a small matrix that definitely fits in RAM.
    """
    n = 500
    dtype = np.float64
    
    a_np = np.random.rand(n, n).astype(dtype)
    b_np = np.random.rand(n, n).astype(dtype)
    
    a = pycauset.matrix(a_np, storage="ram")
    b = pycauset.matrix(b_np, storage="ram")
    
    # Perform matmul
    c = pycauset.matmul(a, b)
    c_np = a_np @ b_np
    
    np.testing.assert_allclose(pycauset.to_numpy(c, allow_huge=True), c_np, rtol=1e-5, atol=1e-8)

def test_bit_matrix_consistency():
    """
    Test boolean/bit matrix IO and alignment.
    """
    rows, cols = 1024, 1024 # Multiple of 64 for alignment checks
    
    data = np.random.randint(0, 2, size=(rows, cols)).astype(bool)
    
    mat = pycauset.matrix(data, storage="disk")
    result = pycauset.to_numpy(mat, allow_huge=True)
    
    np.testing.assert_array_equal(data, result)
if __name__ == "__main__":
    # Manual run if executed as script
    try:
        test_io_roundtrip_consistency(np.float64)
        print("test_io_roundtrip_consistency passed")
        test_large_file_consistency()
        print("test_large_file_consistency passed")
        test_direct_path_consistency()
        print("test_direct_path_consistency passed")
        test_bit_matrix_consistency()
        print("test_bit_matrix_consistency passed")
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
