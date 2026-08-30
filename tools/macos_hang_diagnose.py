"""Isolate the macOS CI hang at test_io_consistency.py::test_direct_path_consistency.

Prints a progress line between every step so the last line in the CI log pins
the exact operation that stalls. Run with `python -u`.
"""
import sys
import os
import numpy as np
import pycauset

print("== import ok", flush=True)
print("== threadpool threads:", pycauset.get_num_threads() if hasattr(pycauset, "get_num_threads") else "n/a", flush=True)
print("== memory threshold:", pycauset.get_memory_threshold(), flush=True)


def step(name):
    print("STEP " + name, flush=True)


# --- Reproduce test_large_file_consistency (passes on macOS) ---
step("large: build numpy 3500x3500")
n = 3500
data = np.full((n, n), 3.14159, dtype=np.float64)
step("large: matrix(storage='disk')")
mat = pycauset.matrix(data, storage="disk")
step("large: backing = " + str(mat.get_backing_file()))
step("large: to_numpy(allow_huge=True)")
result = pycauset.to_numpy(mat, allow_huge=True)
step("large: assert")
np.testing.assert_array_equal(data, result)
step("large: del locals")
del data, mat, result
step("large: done")

# --- Reproduce test_direct_path_consistency (hangs on macOS) ---
step("direct: numpy rand 500x500")
a_np = np.random.rand(500, 500).astype(np.float64)
b_np = np.random.rand(500, 500).astype(np.float64)
step("direct: matrix a (storage='ram')")
a = pycauset.matrix(a_np, storage="ram")
step("direct: matrix b (storage='ram')")
b = pycauset.matrix(b_np, storage="ram")
step("direct: matmul(a, b)")
c = pycauset.matmul(a, b)
step("direct: numpy reference matmul")
c_np = a_np @ b_np
step("direct: to_numpy(c, allow_huge=True)")
c_out = pycauset.to_numpy(c, allow_huge=True)
step("direct: assert_allclose")
np.testing.assert_allclose(c_out, c_np, rtol=1e-5, atol=1e-8)
step("direct: done")

print("== ALL DONE", flush=True)
