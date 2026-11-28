"""Out-of-core / RAM-limit demonstration.

Shows the key difference between NumPy and PyCauset on large matrices:

- NumPy must hold the whole array in RAM, so the maximum square float64 matrix on a
  16 GB machine is about 45000 x 45000 (16.2 GB). Beyond that, NumPy raises MemoryError.
- PyCauset memory-maps to disk beyond a RAM budget (`set_memory_threshold`): the matrix's pages
  are file pages, which the OS can drop and re-read, not anonymous memory. The demonstration
  measures both (RssAnon / RssFile); a "used memory" meter counts the file pages too, but they
  are reclaimable page cache.

Usage: python benchmarks/bench_ram.py
"""
from __future__ import annotations


import pycauset as pc


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return float("nan")


def _rss_mb() -> tuple[float, float]:
    """(anonymous, file-backed) resident memory in MiB: /proc on Linux, else psutil's USS as the
    anonymous part (nan without either)."""
    try:
        fields = dict(line.split(":", 1) for line in open("/proc/self/status"))
        kb = lambda k: float(fields[k].split()[0]) / 1024  # noqa: E731
        return kb("RssAnon"), kb("RssFile")
    except (OSError, KeyError):
        try:
            import psutil

            return psutil.Process().memory_full_info().uss / 2**20, float("nan")
        except Exception:
            return float("nan"), float("nan")


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
    anon0, file0 = _rss_mb()
    m = pc.ones((n, n))
    bf = m.get_backing_file()
    on_disk = bf != ":memory:" and bf is not None
    print(f"  backing: {'disk (' + bf + ')' if on_disk else bf}")

    # A NumPy array of the same size would require the full amount in RAM.
    np_ram = size_gb
    print(f"  NumPy would need {np_ram:.2f} GB of anonymous memory for the same array.")

    # Verify operations still work on the disk-backed matrix (sum reads every page).
    print(f"  trace(ones) = {pc.trace(m)} (expect {n}), sum = {pc.sum(m)} (expect {n * n})")
    anon1, file1 = _rss_mb()
    print(f"  measured: anonymous memory +{anon1 - anon0:.0f} MiB; file pages resident +{file1 - file0:.0f} MiB")
    print("  (the file pages are page cache of the backing file: the OS drops them under pressure")
    print("   and reads them back, so the matrix is not bounded by RAM; anonymous memory stays small)")

    pc.set_memory_threshold(orig)


if __name__ == "__main__":
    main()
