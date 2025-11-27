import pycauset
import os
import numpy as np
import warnings


def test_python_interface():
    print("--- Basic Interface Test ---")
    print("Creating CausalMatrix(N=10)...")
    mat = pycauset.CausalMatrix(10, "py_test.bin")

    print("Setting bits using [i, j] syntax...")
    mat[0, 1] = True
    mat[1, 2] = True

    print(f"Value at (0, 1): {mat[0, 1]}")

    print("\n--- Numpy Interop Test ---")
    arr = np.zeros((5, 5), dtype=bool)
    arr[0, 1] = True
    arr[1, 2] = True

    print("Creating CausalMatrix from Numpy array...")
    mat_np = pycauset.CausalMatrix(arr, "py_np_test.bin")
    print(f"Value at (0, 1): {mat_np[0, 1]}")

    print("\n--- Random Matrix Test ---")
    print("Generating random matrix (N=100, density=0.5)...")
    mat_rnd = pycauset.CausalMatrix.random(100, 0.5, "py_rnd.bin")
    print(f"Shape: {mat_rnd.shape}")

    print("\n--- Guardrails Test ---")
    print("Attempting to set diagonal element (should warn)...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mat[0, 0] = True
        if w:
            print(f"Caught warning: {w[-1].message}")
        else:
            print("FAILURE: No warning caught!")

    print("\n--- Advanced Indexing Test ---")
    rows = np.array([0, 1, 2])
    cols = np.array([5, 6, 7])
    vals = np.array([True, True, True])

    print("Batch setting elements...")
    mat[rows, cols] = vals
    print(f"Value at (0, 5): {mat[0, 5]}")

    for f in ["py_test.bin", "py_np_test.bin", "py_rnd.bin"]:
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == "__main__":
    test_python_interface()
