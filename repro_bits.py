import pycauset
import numpy as np
import sys

def test_bit_repro():
    rows, cols = 1024, 1024
    
    # Use deterministic data
    np.random.seed(42)
    data = np.random.randint(0, 2, size=(rows, cols)).astype(bool)
    
    mat = pycauset.matrix(data, storage="disk")
    result = pycauset.to_numpy(mat, allow_huge=True)
    
    if np.array_equal(data, result):
        print("Success")
        return

    print("Mismatch found")
    diff = data != result
    indices = np.where(diff)
    print(f"Total discrepancies: {np.sum(diff)}")
    
    for i in range(min(20, len(indices[0]))):
        r, c = indices[0][i], indices[1][i]
        print(f"Mismatch at ({r}, {c}): Expected {data[r,c]}, Got {result[r,c]}")

    # Analyze periodicity
    cols_diff = indices[1]
    unique_cols = np.unique(cols_diff)
    print(f"Unique failing columns (first 20): {unique_cols[:20]}")
    
    # Check if whole columns are bad
    # Or specific bits in words.
    
    # Modulo 64 analysis
    mods = cols_diff % 64
    unique_mods = np.unique(mods)
    print(f"Failing indices % 64: {unique_mods}")

    # Modulo 16 analysis (SIMD width)
    mods16 = cols_diff % 16
    unique_mods16 = np.unique(mods16)
    print(f"Failing indices % 16: {unique_mods16}")

if __name__ == "__main__":
    test_bit_repro()
