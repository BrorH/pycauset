import os
import ctypes
import sys

package_dir = os.path.abspath("python/pycauset")
os.add_dll_directory(package_dir)

dlls = [
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudart64_12.dll",
    "cusolver64_11.dll",
    "cusparse64_12.dll",
    "nvJitLink_120_0.dll",
    "_pycauset.dll",
    "pycauset_cuda.dll"
]

print(f"Loading DLLs from {package_dir}...")

for dll in dlls:
    path = os.path.join(package_dir, dll)
    print(f"Loading {dll}...", end=" ")
    try:
        ctypes.CDLL(path)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
