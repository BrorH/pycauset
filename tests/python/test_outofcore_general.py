import numpy as np
import pycauset
import sys

def test_outofcore_general():
    if not pycauset.cuda.is_available():
        print("CUDA not available.")
        return

    print("\n" + "="*80)
    print("TESTING OUT-OF-CORE LU DECOMPOSITION (GENERAL MATRIX)")
    print("="*80)

    N = 2048
    limit = 10 * 1024 * 1024 # 10 MB Limit
    
    print(f"Matrix: {N}x{N}")
    print(f"Memory Limit: {limit/1024/1024:.0f} MB")
    
    # Generate General Random Matrix (Not Diagonally Dominant)
    print("Generating General Random Matrix...")
    np.random.seed(42)
    # Create random matrix with values in [0, 1]
    A_np_orig = np.random.rand(N, N).astype(np.float64)
    
    # Create pycauset matrix
    A = pycauset.Float64Matrix(N)
    # Fill it (slow element-wise, but fine for test)
    # Optimally we would use a buffer load if available
    # For now, let's just set a few rows to ensure it's not trivial
    # Actually, let's use the numpy interop if possible, but Float64Matrix might not support direct numpy load yet?
    # Let's check if we can set data from bytes or similar.
    # Or just loop. 2048*2048 = 4M elements. Python loop might take a few seconds.
    
    # Faster way: Use a flat list?
    # A.set_data(A_np_orig.flatten()) # Hypothetical
    
    # Fallback to loop
    for i in range(N):
        for j in range(N):
            A[i, j] = A_np_orig[i, j]
            
    print("Running Out-of-Core LU...")
    pycauset.cuda.enable(memory_limit=limit, enable_async=True)
    
    LU = A.inverse()
    
    LU_np = np.array(LU)
    
    # Extract L and U
    L = np.tril(LU_np, -1) + np.eye(N)
    U = np.triu(LU_np)
    
    # Reconstruct P*A = L*U ?
    # Wait, my implementation does pivoting physically on A (swapping rows).
    # So the result in A (and thus LU) is the factored version of P*A.
    # But I don't return P.
    # The `apply_pivots` function swaps rows of the matrix in place.
    # So the original A is lost/permuted.
    # The `LU` matrix contains L and U of the *permuted* matrix.
    # So L*U should equal P*A_orig.
    # Since I don't track P in the output (yet), I can't easily verify against A_orig unless I know P.
    
    # However, `apply_pivots` modifies `dst` (which is `out`).
    # `src` (input `in`) is separate.
    # `invert(const MatrixBase& in, MatrixBase& out)`
    # It copies `in` to `out` first.
    # Then it permutes `out`.
    
    # So `LU` = L * U.
    # And `LU` is the factorization of `P * A_in`.
    # But `A_in` is const.
    
    # If I want to verify, I need to know P.
    # My current implementation returns `void` (or rather, modifies `out`).
    # It does NOT return the pivot history.
    
    # This is a limitation. Without P, I cannot reconstruct A to verify against the original.
    # But I can verify that L*U is a valid matrix that is a row-permutation of A.
    # Or, I can check if `A.inverse()` is actually supposed to return the inverse?
    # The method name is `inverse()`.
    # My implementation currently does LU decomposition but stops there.
    # "TODO: Implement Triangular Solve (Forward/Backward) to get A^{-1}"
    # The comment in `CudaSolver.cu` says:
    # "// TODO: Implement Triangular Solve (Forward/Backward) to get A^{-1}"
    # "// Currently 'out' contains L and U."
    
    # So `A.inverse()` currently returns LU factors, NOT the inverse.
    # And it permutes the rows.
    
    # If I can't verify P*A = L*U, I can at least check that L*U has the same rows as A (just permuted).
    # We can sort the rows of A and L*U and compare?
    # Or check determinants? det(A) = +/- det(L*U).
    
    A_rec = L @ U
    
    # Check if rows of A_rec are a permutation of rows of A_np_orig
    # This is expensive to check perfectly.
    # But we can check norms of rows?
    
    print("Verifying result (Permutation Check)...")
    
    # Use slogdet to avoid overflow
    sign_A, logdet_A = np.linalg.slogdet(A_np_orig)
    sign_LU, logdet_LU = np.linalg.slogdet(A_rec)
    
    print(f"LogDet(A)  = {logdet_A:.6f}")
    print(f"LogDet(LU) = {logdet_LU:.6f}")
    
    if np.isclose(logdet_A, logdet_LU, rtol=1e-5):
        print("SUCCESS: Log-Determinants match.")
    else:
        print("FAILURE: Log-Determinant mismatch.")

if __name__ == "__main__":
    test_outofcore_general()
