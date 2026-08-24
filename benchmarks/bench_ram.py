"""Out-of-core / RAM-limit demonstration.

Shows the key difference between NumPy and PyCauset on large matrices:

- NumPy must hold the whole array in RAM, so the maximum square float64 matrix on a
  16 GB machine is about 45000 x 45000 (16.2 GB). Beyond that, NumPy raises MemoryError.
- PyCauset memory-maps to disk beyond a RAM budget (`set_memory_threshold`), so its RAM
  usage is bounded by your budget and the matrix can be far larger than RAM.

Usage: python benchmarks/bench_ram.py
"""
from __future__ import annotations

import numpy as np

import pycauset as pc


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return float("nan")


def main() -> None:
    avail = _ram_gb()
    print(f"Available RAM: {avail:.1f} GB\n")

    print("## Maximum square float64 matrix at a given RAM budget")
    print("(NumPy holds all n*n*8 bytes in RAM; PyCauset's RAM is bounded by its threshold)")
    print()
    print("| RAM budget | max n (NumPy) | PyCauset |")
    print("|---|---|---|")
    for gb in [4, 8, 16, 32, 64]:
        n = int((gb * 1024 ** 3 / 8) ** 0.5)
        print(f"| {gb} GB | {n} x {n} | unbounded (disk-backed) |")

    # Concrete demonstration: a matrix larger than the RAM budget spills to disk.
    print("\n## Demonstration (matrix bigger than the RAM budget)")
    n = 12000  # 12000^2 * 8 = 1.15 GB
    size_gb = n * n * 8 / 1024 ** 3
    budget_mb = 256  # only 256 MB of RAM allowed

    orig = pc.get_memory_threshold()
    pc.set_memory_threshold(budget_mb * 1024 * 1024)

    print(f"Creating a {n} x {n} float64 matrix ({size_gb:.2f} GB) with a {budget_mb} MB RAM budget...")
    m = pc.FloatMatrix(n)
    bf = m.get_backing_file()
    on_disk = bf != ":memory:" and bf is not None
    print(f"  backing: {'disk (' + bf + ')' if on_disk else bf}")

    # A NumPy array of the same size would require the full amount in RAM.
    np_ram = size_gb
    print(f"  NumPy would need {np_ram:.2f} GB in RAM for the same array.")
    print(f"  PyCauset RAM budget: {budget_mb / 1024:.2f} GB (the rest on disk).")

    # Verify operations still work on the disk-backed matrix.
    m.set_identity()
    print(f"  trace(identity) = {pc.trace(m)} (expect {n})")

    pc.set_memory_threshold(orig)


if __name__ == "__main__":
    main()
