import os
import time
from pathlib import Path

import psutil
import pycauset


def _format_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"


def main() -> int:
    print("[R1_GPU] 100GB Matrix Stress Test")

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.cwd()))
    print(f"RAM total: {_format_bytes(vm.total)} | available: {_format_bytes(vm.available)}")
    print(f"Disk total: {_format_bytes(disk.total)} | free: {_format_bytes(disk.free)}")

    stress_dir = Path("C:/Users/ireal/Documents/pycauset/tmp_stress_100gb")
    stress_dir.mkdir(parents=True, exist_ok=True)
    pycauset.set_backing_dir(stress_dir)

    # Force disk-backed storage even on large-memory machines.
    old_threshold = pycauset.get_memory_threshold()
    try:
        pycauset.set_memory_threshold(256 * 1024 * 1024)  # 256 MB

        target_bytes = 100 * 1024 ** 3  # 100 GiB
        n = int((target_bytes / 8) ** 0.5)
        size_bytes = n * n * 8
        if size_bytes < target_bytes:
            n += 1
            size_bytes = n * n * 8

        print(f"Target: 100 GiB, using n={n}")
        print(f"Matrix size: {_format_bytes(size_bytes)} (float64)")

        start = time.time()
        mat = pycauset.FloatMatrix(n)
        create_time = time.time() - start
        print(f"Create time: {create_time:.2f}s")

        backing = mat.get_backing_file()
        print(f"Backing file: {backing}")

        # Touch a few elements (no full fill) to validate access on disk-backed storage.
        mat.set(0, 0, 1.0)
        mat.set(n - 1, n - 1, 2.0)
        mat.set(n // 2, n // 2, 3.0)

        v0 = mat.get(0, 0)
        v1 = mat.get(n - 1, n - 1)
        v2 = mat.get(n // 2, n // 2)
        print(f"Sample reads: (0,0)={v0}, (n-1,n-1)={v1}, (mid,mid)={v2}")

        if backing and backing != ":memory:" and Path(backing).exists():
            print(f"Backing file size: {_format_bytes(Path(backing).stat().st_size)}")

        mat.close()
        return 0
    finally:
        # Restore threshold
        pycauset.set_memory_threshold(old_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
