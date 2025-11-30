
import sys
import os

# Add python directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'python'))

import pycauset

try:
    print("Attempting to create CausalMatrix from list of lists...")
    data = [[0, 1], [0, 0]]
    C = pycauset.CausalMatrix(data)
    print("Success!")
    print(C)
except Exception as e:
    print(f"Failed: {e}")

try:
    print("\nAttempting to create TriangularBitMatrix from list of lists...")
    data = [[0, 1], [0, 0]]
    C = pycauset.TriangularBitMatrix(data)
    print("Success!")
    print(C)
except Exception as e:
    print(f"Failed: {e}")
