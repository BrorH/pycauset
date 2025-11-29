import tempfile
from pathlib import Path

import numpy as np
import warnings

import pycauset


def test_python_interface():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        def backing(name: str) -> Path:
            return tmpdir_path / name

        print("--- Basic Interface Test ---")
        print("Creating causalmatrix(N=10)...")
        mat = pycauset.causalmatrix(10, backing("py_test"))

        print("Setting bits using [i, j] syntax...")
        mat[0, 1] = True
        mat[1, 2] = True

        print(f"Value at (0, 1): {mat[0, 1]}")

        print("\n--- Numpy Interop Test ---")
        arr = np.zeros((5, 5), dtype=bool)
        arr[0, 1] = True
        arr[1, 2] = True

        print("Creating causalmatrix from NumPy array...")
        mat_np = pycauset.causalmatrix(arr, backing("py_np_test"))
        print(f"Value at (0, 1): {mat_np[0, 1]}")

        print("\n--- Random Matrix Test ---")
        print("Generating random matrix (N=100, density=0.5)...")
        mat_rnd = pycauset.causalmatrix.random(100, 0.5, backing("py_rnd"))
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

        mat.close()
        mat_np.close()
        mat_rnd.close()


if __name__ == "__main__":
    test_python_interface()
