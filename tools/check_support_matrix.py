"""Dev tool: execute the support matrix and fail fast on regressions.

Run from repo root:
  - `python tools/check_support_matrix.py`

This is intentionally lightweight and only enforces what the internal support
matrix declares as supported.
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings


def _repo_import():
    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    sys.path.insert(0, str(python_dir))
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _repo_import()

    import pycauset
    from pycauset._internal.support_matrix import SUPPORTED

    failures: list[str] = []

    def _values_matrix(dtype: str):
        if dtype == "bit":
            return [[1, 0], [1, 1]]
        if dtype in ("int16", "int32"):
            return [[2, -3], [4, 5]]
        if dtype in ("float16", "float32", "float64"):
            return [[1.25, -2.5], [3.0, 0.5]]
        return [[1 + 2j, 3 - 4j], [-5 + 0.5j, 0 + 6j]]

    def _values_vector(dtype: str):
        if dtype == "bit":
            return [1, 0]
        if dtype in ("int16", "int32"):
            return [2, -3]
        if dtype in ("float16", "float32", "float64"):
            return [1.25, -2.5]
        return [1 + 2j, -3 + 0.5j]

    def make_matrix(dtype: str):
        m = pycauset.empty((2, 2), dtype=dtype)
        vals = _values_matrix(dtype)
        m[0, 0] = vals[0][0]
        m[0, 1] = vals[0][1]
        m[1, 0] = vals[1][0]
        m[1, 1] = vals[1][1]
        return m

    def make_vector(dtype: str):
        v = pycauset.empty(2, dtype=dtype)
        vals = _values_vector(dtype)
        v[0] = vals[0]
        v[1] = vals[1]
        return v

    for case in SUPPORTED:
        try:
            a_dtype = case.a_dtype
            b_dtype = case.b_dtype or case.a_dtype

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if case.kind == "matrix":
                    a = make_matrix(a_dtype)
                    b = make_matrix(b_dtype)
                    out = None
                    try:
                        if case.op == "add":
                            out = a + b
                        elif case.op == "sub":
                            out = a - b
                        elif case.op == "mul":
                            out = a * b
                        elif case.op == "matmul":
                            out = a @ b
                        elif case.op == "H":
                            out = a.H
                        elif case.op == "conj":
                            out = a.conj()
                        else:
                            raise ValueError(f"Unknown op: {case.op}")

                        _ = __import__("numpy").array(out)
                    finally:
                        for obj in (out, b, a):
                            if obj is not None and hasattr(obj, "close"):
                                try:
                                    obj.close()
                                except Exception:
                                    pass

                elif case.kind == "vector":
                    a = make_vector(a_dtype)
                    b = make_vector(b_dtype)
                    out = None
                    try:
                        if case.op == "add":
                            out = a + b
                        elif case.op == "sub":
                            out = a - b
                        elif case.op == "dot":
                            _ = a.dot(b)
                        elif case.op == "outer":
                            out = a @ b.T
                        elif case.op == "H":
                            out = a.H
                        elif case.op == "conj":
                            out = a.conj()
                        else:
                            raise ValueError(f"Unknown op: {case.op}")

                        if out is not None:
                            _ = __import__("numpy").array(out)
                    finally:
                        for obj in (out, b, a):
                            if obj is not None and hasattr(obj, "close"):
                                try:
                                    obj.close()
                                except Exception:
                                    pass

                elif case.kind == "matvec":
                    m = make_matrix(a_dtype)
                    v = make_vector(b_dtype)
                    out = None
                    try:
                        out = m @ v
                        _ = __import__("numpy").array(out)
                    finally:
                        for obj in (out, v, m):
                            if obj is not None and hasattr(obj, "close"):
                                try:
                                    obj.close()
                                except Exception:
                                    pass

                elif case.kind == "vecmat":
                    v = make_vector(a_dtype)
                    m = make_matrix(b_dtype)
                    out = None
                    try:
                        out = v @ m
                        _ = __import__("numpy").array(out)
                    finally:
                        for obj in (out, m, v):
                            if obj is not None and hasattr(obj, "close"):
                                try:
                                    obj.close()
                                except Exception:
                                    pass

                else:
                    raise ValueError(f"Unknown kind: {case.kind}")

        except Exception as exc:
            failures.append(f"{case.kind}:{case.op}:{a_dtype},{b_dtype} -> {type(exc).__name__}: {exc}")

    if failures:
        print("FAIL: support matrix regressions:")
        for f in failures:
            print("  -", f)
        return 1

    print("OK: support matrix cases all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
