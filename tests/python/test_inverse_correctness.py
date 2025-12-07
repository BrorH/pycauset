import numpy as np
import pycauset
import sys
import time

def test_inverse_correctness():
    if not pycauset.cuda.is_available():
        print("CUDA not available. Skipping test.")
        return

    print("\n" + "="*80)
    print("TESTING OUT-OF-CORE INVERSE CORRECTNESS")
    print("="*80)

    sizes = [512, 1024, 2048]
    limit = 10 * 1024 * 1024 # 10 MB Limit

    for N in sizes:
        print(f"\nTesting Matrix Size: {N}x{N}")
        
        # Generate General Random Matrix
        np.random.seed(42)
        A_np = np.random.rand(N, N).astype(np.float64)
        
        # Ensure it's well-conditioned enough to invert without huge errors
        # Add identity to diagonal to help condition number slightly, though random is usually fine
        A_np += np.eye(N) * 2.0
        
        print("Creating PyCauset Matrix...")
        A = pycauset.Float64Matrix(N)
        # Fill data
        for i in range(N):
            for j in range(N):
                A[i, j] = A_np[i, j]
                
        print("Computing Inverse (Out-of-Core)...")
        pycauset.cuda.enable(memory_limit=limit, enable_async=True)
        
        start_time = time.time()
        A_inv = A.inverse()
        end_time = time.time()
        print(f"Inversion Time: {end_time - start_time:.4f} s")
        
        A_inv_np = np.array(A_inv)
        
        print("Verifying A * A_inv = I...")
        # Compute product
        I_rec = A_np @ A_inv_np
        
        # Check against Identity
        I_ref = np.eye(N)
        diff = np.abs(I_rec - I_ref)
        max_diff = np.max(diff)
        
        print(f"Max Error |A * A_inv - I|: {max_diff:.6e}")
        
        if max_diff < 1e-8:
            print("SUCCESS: Inverse is correct.")
        else:
            print("FAILURE: Inverse error too high.")
            # Print some debug info
            print("Top-left 5x5 of A * A_inv:")
            print(I_rec[:5, :5])

if __name__ == "__main__":
    test_inverse_correctness()
