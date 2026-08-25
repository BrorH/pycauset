import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pycauset


def _mb_per_s(bytes_count: float, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return (bytes_count / 1_000_000.0) / seconds


def run_io_smoke(size: int | None = None, repeats: int = 1) -> dict[str, float]:
    side = size or int(os.environ.get("PYCAUSET_IO_BENCH_SIZE", "512"))
    reps = max(1, repeats)
    payload_bytes = float(side * side * 4)  # float32 default

    arr = np.ones((side, side), dtype=np.float32)
    m = pycauset.matrix(arr)

    metrics: dict[str, float] = {"size": float(side), "bytes": payload_bytes}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "io_smoke.pycauset"

        save_times = []
        load_times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            pycauset.save(m, path)
            save_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            loaded = pycauset.load(path)
            load_times.append(time.perf_counter() - t1)
            loaded.close()

        metrics["save_s"] = float(sum(save_times) / len(save_times))
        metrics["load_s"] = float(sum(load_times) / len(load_times))
        metrics["save_mb_s"] = _mb_per_s(payload_bytes, metrics["save_s"])
        metrics["load_mb_s"] = _mb_per_s(payload_bytes, metrics["load_s"])

    m.close()
    return metrics


if __name__ == "__main__":
    result = run_io_smoke()
    print(json.dumps(result, indent=2))
