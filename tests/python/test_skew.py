import unittest

raise unittest.SkipTest(
    "Skew eigenvalue solver was removed along with the legacy complex/eigen subsystem."
)

import sys
import os
import numpy as np
import time

# Add the python directory to the path so we can import pycauset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

import pycauset

def test_skew_symmetric_solver():
    print("Testing Skew-Symmetric Eigenvalue Solver...")
    
    N = 100
    k = 10
    
    # 1. Create a random matrix
    np.random.seed(42)
    M = np.random.rand(N, N)
    
    # 2. Make it skew-symmetric: A = M - M^T
    # A^T = (M - M^T)^T = M^T - M = -(M - M^T) = -A
    A_np = M - M.T
    
    # 3. Create PyCauset matrix
    # We use Float64 for accuracy in this test
    A_pc = pycauset.matrix(A_np)
    
    print(f"Matrix size: {N}x{N}")
    print(f"Computing {k} eigenvalues...")
    
    # 4. Run the solver
    start_time = time.time()
    # The solver returns complex eigenvalues
    evals = pycauset.eigvals_skew(A_pc, k)
    end_time = time.time()
    
    print(f"Skew Solver time: {end_time - start_time:.4f}s")
    
    print("Eigenvalues returned:")
    for i in range(evals.size()):
        e = evals.get(i)
        print(f"  {e}")
        
    # 5. Verification
    
    # Check 1: Purely imaginary
    # The real part should be negligible (machine epsilon)
    max_real_part = 0.0
    for i in range(evals.size()):
        e = evals.get(i)
        if abs(e.real) > max_real_part:
            max_real_part = abs(e.real)
            
    print(f"Max real part: {max_real_part}")
    if max_real_part > 1e-10:
        print("FAIL: Eigenvalues are not purely imaginary!")
        sys.exit(1)
        
    # Check 2: Compare with Numpy
    # Numpy returns all N eigenvalues. We need the k largest magnitude ones.
    print("Comparing with Numpy...")
    evals_np = np.linalg.eigvals(A_np)
    
    # Sort numpy eigenvalues by magnitude (descending)
    evals_np_sorted = sorted(evals_np, key=lambda x: abs(x), reverse=True)
    
    print("\nTop 5 Numpy eigenvalues (magnitude):")
    for i in range(min(k, 5)):
        print(f"  {evals_np_sorted[i]}")
        
    # Compare magnitudes
    # We allow some tolerance because Arnoldi is an iterative method
    tolerance = 1e-5
    passed = True
    
    print("\nComparison (Skew vs Numpy):")
    for i in range(k):
        mag_pc = abs(evals.get(i))
        mag_np = abs(evals_np_sorted[i])
        diff = abs(mag_pc - mag_np)
        
        if diff > tolerance:
            print(f"Mismatch at index {i}: PyCauset={mag_pc}, Numpy={mag_np}, Diff={diff}")
            passed = False
            
    if passed:
        print("\nSUCCESS: Skew Eigenvalues match Numpy results!")
    else:
        print("\nFAIL: Skew Eigenvalues do not match Numpy results.")
        sys.exit(1)

if __name__ == "__main__":
    test_skew_symmetric_solver()
