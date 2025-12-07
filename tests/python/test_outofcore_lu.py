import numpy as np
import pycauset
import sys

def test_outofcore_lu():
    if not pycauset.cuda.is_available():
        print("CUDA not available.")
        return

    print("\n" + "="*80)
    print("TESTING OUT-OF-CORE LU DECOMPOSITION")
    print("="*80)

    # Setup: Medium Matrix (large enough to trigger blocked logic if we lower limit)
    N = 2048
    limit = 10 * 1024 * 1024 # 10 MB Limit -> Forces blocking (N*N*8 = 32MB)
    
    print(f"Matrix: {N}x{N}")
    print(f"Memory Limit: {limit/1024/1024:.0f} MB")
    
    # Generate Diagonally Dominant Matrix (Float64)
    # pycauset.Float64Matrix.random(N) creates a diagonally dominant matrix
    print("Generating Float64Matrix...")
    A = pycauset.Float64Matrix.random(N)
    
    print(f"Matrix Type: {type(A)}")
    
    print("Running Out-of-Core LU (via invert)...")
    pycauset.cuda.enable(memory_limit=limit, enable_async=True)
    
    # Currently 'invert' returns LU in the output matrix
    # It does NOT return the inverse yet.
    LU = A.inverse()
    # pycauset.cuda.inverse(A, LU)
    
    LU_np = np.array(LU)
    
    # Extract L and U
    L = np.tril(LU_np, -1) + np.eye(N)
    U = np.triu(LU_np)
    
    # Reconstruct A
    A_rec = L @ U
    
    # Convert original A to numpy for comparison
    # Note: This might be slow due to element-wise access if buffer protocol isn't optimized
    print("Verifying result...")
    A_np = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            A_np[i, j] = A[i, j]
            
    # Check Error
    diff = np.abs(A_np - A_rec)
    max_diff = np.max(diff)
    print(f"Max Reconstruction Error: {max_diff:.6e}")
    
    if max_diff < 1e-10:
        print("SUCCESS: LU Decomposition is correct.")
    else:
        print("FAILURE: Reconstruction error too high.")

if __name__ == "__main__":
    test_outofcore_lu()
