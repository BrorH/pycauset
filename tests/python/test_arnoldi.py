import sys
import os
import time
import numpy as np
import pytest

# Add the python directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

import pycauset

try:
    import pycauset._pycauset as _native
except Exception:  # pragma: no cover
    _native = None


_HAVE_EIG = (
    _native is not None
    and hasattr(_native, "eigvals")
    and hasattr(_native, "eigvals_arnoldi")
)


@pytest.mark.skipif(
    not _HAVE_EIG,
    reason="pycauset eigensolver APIs (eigvals/eigvals_arnoldi) are not available in this build",
)
def test_arnoldi_small():
    print("Testing Arnoldi on small matrix (N=100)...")
    N = 100
    
    m = pycauset.FloatMatrix(N)
    
    # Fill with random data
    # This is slow from python loop.
    # But for N=100 it's fine.
    np.random.seed(42)
    arr = np.random.rand(N, N)
    
    for i in range(N):
        for j in range(N):
            m.set(i, j, arr[i, j])
            
    print("Matrix created.")
    
    # Run Dense Solver
    t0 = time.time()
    evals_dense = pycauset.eigvals(m)
    t1 = time.time()
    print(f"Dense solver time: {t1 - t0:.4f}s")
    
    # Run Arnoldi Solver (k=10)
    k = 10
    t0 = time.time()
    evals_arnoldi = pycauset.eigvals_arnoldi(m, k, 30, 1e-6)
    t1 = time.time()
    print(f"Arnoldi solver time: {t1 - t0:.4f}s")
    
    # Compare top k eigenvalues
    # Sort dense eigenvalues by magnitude
    dense_vals = []
    for i in range(N):
        c = evals_dense.get(i)
        dense_vals.append(complex(c.real, c.imag))
        
    dense_vals.sort(key=abs, reverse=True)
    
    arnoldi_vals = []
    # Arnoldi returns m eigenvalues (where m is subspace size, e.g. 30)
    # We need to check how many it returned.
    n_arnoldi = evals_arnoldi.size()
    for i in range(n_arnoldi):
        c = evals_arnoldi.get(i)
        arnoldi_vals.append(complex(c.real, c.imag))
        
    arnoldi_vals.sort(key=abs, reverse=True)
    
    print("Top 5 Dense vs Arnoldi:")
    for i in range(5):
        print(f"Dense:   {dense_vals[i]:.6f}")
        print(f"Arnoldi: {arnoldi_vals[i]:.6f}")
        
    # Check error
    err = 0.0
    for i in range(min(k, len(arnoldi_vals))):
        err += abs(dense_vals[i] - arnoldi_vals[i])
    print(f"Total error for top {k}: {err}")

@pytest.mark.skipif(
    not _HAVE_EIG,
    reason="pycauset eigensolver APIs (eigvals/eigvals_arnoldi) are not available in this build",
)
@pytest.mark.skipif(
    os.environ.get("PYCAUSET_RUN_SLOW") not in {"1", "true", "TRUE", "yes", "YES"},
    reason="slow benchmark-style test; set PYCAUSET_RUN_SLOW=1 to enable",
)
def test_arnoldi_large():
    print("\nTesting Arnoldi on larger matrix (N=1000)...")
    N = 1000
    m = pycauset.FloatMatrix(N)
    
    # Fill with random data (slow in python, but bearable for 1000x1000 = 1M elements)
    # Actually, let's use a simpler fill to be faster
    # Or just rely on uninitialized garbage? No, that's bad.
    # Let's fill diagonal to make it easy? No, we want dense.
    # We can use `m.random()` if available?
    # `FloatMatrix` doesn't seem to have `random`. `TriangularBitMatrix` does.
    
    # We'll just fill it.
    np.random.seed(42)
    arr = np.random.rand(N, N)
    for i in range(N):
        for j in range(N):
            m.set(i, j, arr[i, j])
            
    print("Matrix created.")
    
    # Run Dense Solver
    t0 = time.time()
    evals_dense = pycauset.eigvals(m)
    t1 = time.time()
    print(f"Dense solver time: {t1 - t0:.4f}s")
    
    # Run Arnoldi Solver (k=10)
    k = 10
    t0 = time.time()
    evals_arnoldi = pycauset.eigvals_arnoldi(m, k, 50, 1e-6)
    t1 = time.time()
    print(f"Arnoldi solver time: {t1 - t0:.4f}s")

if __name__ == "__main__":
    test_arnoldi_small()
    test_arnoldi_large()
