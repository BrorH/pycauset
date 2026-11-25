"""Atlas timing pass (PLAN C.3): speed of PyCauset against NumPy, per cell.

    python -m tests.atlas.timing [op ...]

For every cell the correctness atlas reports as `ok` and `fast` (RAM, same-dtype operands,
views plain / T / scaled_float / conj), time the PyCauset operation at a reference size and
NumPy's same operation on the same logical array in the same dtype. Lazy results are
switched off while timing (results are computed, not deferred); input views are built
before that, so they are still views. Best of three runs each.

    speed = numpy_time / pycauset_time      (1.0 = as fast as NumPy; target >= 0.9)

Writes review/atlas/TIMING.md (committed) and review/atlas/timing.json (scratch).
Not a test: timings depend on the machine. The table is what "fast" means in numbers.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import statistics
import sys
import time
import warnings
import zlib

import numpy as np

import pycauset as pc
from tests.atlas import atlas as A

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "review", "atlas")
N_QUADRATIC = 1024  # O(n^2) operations: 1M elements
N_CUBIC = 384  # O(n^3) operations
VIEWS = ("plain", "T", "scaled_float", "conj")
SKIP_GROUPS = {"io"}  # export / getitem / save_load: no NumPy operation to compare with
SKIP_OPS = {"qr", "lu"}  # no NumPy value reference in the atlas


def _best(fn, budget=1.0):
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        best = min(best, dt)
        if dt > budget:
            break
    return best


_OPERANDS: dict = {}


def _operand(structure, dtype, view, seed, n, linalg):
    A.N = n
    A.W = n - 2
    if structure in ("vector", "unit_vector") and not linalg:
        A.N = n * n  # vectors: throughput at n^2 elements, not per-call overhead
    key = (structure, dtype, seed, A.N, linalg)
    if key not in _OPERANDS:
        dominant = A._dominant
        if not linalg:  # diagonal dominance (+n) is for linear algebra; it overflows int8 at n = 1024
            A._dominant = lambda a, dtype: a
        try:
            if structure == "vector":  # reference() draws N x N first: draw the vector directly
                a = A._values(dtype, (A.N,), seed)
            elif structure == "unit_vector":
                a = np.zeros(A.N, dtype=A.np_dtype(dtype))
                a[seed % A.N] = 1
            else:
                a = A.reference(structure, dtype, seed)
        finally:
            A._dominant = dominant
        _OPERANDS[key] = (_build_fast(structure, dtype, a, seed), a)
    m, a = _OPERANDS[key]
    pc._native._set_lazy(True)  # views are built as views
    v = A.VIEWS[view][0](m)
    vdt = A._pc_dtype_of(v)
    return v, A.view_reference(view, a, vdt), vdt


def _build_fast(structure, dtype, a, seed):
    """A.build, with structured classes filled natively (n^2 Python setitems cost minutes)."""
    cls = {"triangular": lambda n: pc.TriangularMatrix(n, dtype=dtype, has_diagonal=True),
           "strict_triangular": lambda n: pc.TriangularMatrix(n, dtype=dtype, has_diagonal=False),
           "diagonal": lambda n: pc.DiagonalMatrix(n, dtype=dtype),
           "symmetric": lambda n: pc.SymmetricMatrix(n, dtype=dtype),
           "antisymmetric": lambda n: pc.AntiSymmetricMatrix(n, dtype=dtype),
           "hermitian": lambda n: pc.HermitianMatrix(n, dtype=dtype),
           "skew_hermitian": lambda n: pc.SkewHermitianMatrix(n, dtype=dtype)}.get(structure)
    if cls is None:
        return A.build(structure, dtype, a, seed)
    try:
        m = cls(a.shape[0])
    except Exception as e:  # noqa: BLE001 - the structure x dtype is not offered
        raise A.Unavailable(str(e)) from None
    pc._native._debug_fill(m, np.asarray(a, dtype=np.complex128))
    return m


def _previously_fast() -> set[str]:
    with gzip.open(os.path.join(OUT, "atlas.json.gz"), "rt") as f:
        snap = json.load(f)
    return {k for k, v in snap.items() if v == "ok fast"}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    only = set(argv)
    fast = _previously_fast()
    cells = [c for c in A.enumerate_cells(storages=("ram",))
             if c.key() in fast and c.view in VIEWS and not c.dtype2 and not c.structure2
             and (not only or c.op in only)
             and A.OP_BY_NAME[c.op].group not in SKIP_GROUPS and c.op not in SKIP_OPS
             and not (c.view == "conj" and A.kind(c.dtype) != "c")]
    rows = []
    t0 = time.time()
    lazy_before = pc._native._get_lazy() if hasattr(pc._native, "_get_lazy") else True
    for i, c in enumerate(cells):
        op = A.OP_BY_NAME[c.op]
        n = N_QUADRATIC if op.o_n2 else N_CUBIC
        seed = zlib.crc32("/".join((c.op, c.dtype, c.structure)).encode()) % 1000
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, a, xdt = _operand(c.structure, c.dtype, c.view, seed, n, op.linalg)
                y, b, _ = (_operand(c.structure, c.dtype, c.view, seed + 1, n, op.linalg)
                           if op.arity == 2 else (None, None, ""))
                vref = A._values(xdt if xdt in A.DTYPES else "float64", (a.shape[-1],), seed + 7)
                A.DT[0] = xdt
                A.VEC[x.dtype] = pc.vector(vref, dtype=xdt) if xdt in A.DTYPES else None
                A.VEC_REF[xdt] = vref
                rhs = np.linspace(-1, 1, a.shape[0])
                A.RHS[0], A.RHS_REF[0] = pc.vector(rhs), rhs
                pc._native._set_lazy(False)  # time the work, not a deferred result
                try:
                    with np.errstate(all="ignore"):
                        t_np = _best(lambda: op.ref(a, b))
                except TypeError:  # NumPy has no such operation (bool - bool): nothing to compare
                    rows.append(dict(key=c.key(), op=c.op, dtype=c.dtype, structure=c.structure,
                                     view=c.view, n=n, no_numpy=True))
                    continue
                t_pc = _best(lambda: op.run(x, y))
        except Exception as e:  # noqa: BLE001 - reported, never silent
            rows.append(dict(key=c.key(), op=c.op, dtype=c.dtype, structure=c.structure, view=c.view,
                             n=n, error=f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        finally:
            pc._native._set_lazy(lazy_before)
        rows.append(dict(key=c.key(), op=c.op, dtype=c.dtype, structure=c.structure, view=c.view, n=n,
                         t_pc=t_pc, t_np=t_np, speed=t_np / t_pc if t_pc > 0 else float("inf")))
        if i % 100 == 0:
            print(f"{i}/{len(cells)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    A.N = 16
    A.W = 14
    overhead = _call_overhead()
    with open(os.path.join(OUT, "timing.json"), "w") as f:
        json.dump(rows, f, separators=(",", ":"))
    with open(os.path.join(OUT, "TIMING.md"), "w") as f:
        f.write(render(rows, overhead))
    ok = [r for r in rows if "speed" in r]
    print(f"{len(ok)} timed, {sum(r['speed'] < 0.9 for r in ok)} below 0.9x NumPy, "
          f"{sum('no_numpy' in r for r in rows)} without a NumPy equivalent, {sum('error' in r for r in rows)} errors")


def _call_overhead() -> dict:
    """Per-call cost on 4-element operands (microseconds): the Python and binding layers."""
    x, a = pc.vector(np.ones(4)), np.ones(4)
    m, b = pc.matrix(np.ones((2, 2))), np.ones((2, 2))
    out = {}
    for name, f_pc, f_np in [("vector + vector", lambda: x + x, lambda: a + a),
                             ("matrix + matrix", lambda: m + m, lambda: b + b),
                             ("matrix @ matrix", lambda: m @ m, lambda: b @ b),
                             ("matrix * 2.0", lambda: m * 2.0, lambda: b * 2.0),
                             ("norm(vector)", lambda: pc.norm(x), lambda: np.linalg.norm(a))]:
        reps = 2000
        t_pc = min(_loop(f_pc, reps) for _ in range(3)) / reps * 1e6
        t_np = min(_loop(f_np, reps) for _ in range(3)) / reps * 1e6
        out[name] = (t_pc, t_np)
    return out


def _loop(f, reps):
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return time.perf_counter() - t


def render(rows, overhead=None) -> str:
    ok = [r for r in rows if "speed" in r]
    out = ["# Atlas timing (generated)", "",
           "Generated by `python -m tests.atlas.timing`; design in `review/features/ATLAS.md` §3. "
           "Do not edit. Machine-dependent: compare runs on the same machine.", "",
           f"speed = NumPy time / PyCauset time on the same logical data and dtype (1.0 = parity, "
           f"target ≥ 0.9). Sizes: n = {N_QUADRATIC} for O(n²) operations, n = {N_CUBIC} for O(n³). "
           f"Cells: the atlas's `fast` cells, RAM, views {', '.join(VIEWS)}.", "",
           f"{len(ok)} cells timed; {sum(r['speed'] < 0.9 for r in ok)} below 0.9; "
           f"{sum('no_numpy' in r for r in rows)} without a NumPy equivalent (e.g. bool - bool); "
           f"{sum('error' in r for r in rows)} errors.", "",
           "## Per operation", "",
           "| op | cells | median speed | worst | below 0.9 |", "|---|---|---|---|---|"]
    by_op = collections.defaultdict(list)
    for r in ok:
        by_op[r["op"]].append(r)
    for op, rs in sorted(by_op.items(), key=lambda kv: statistics.median(r["speed"] for r in kv[1])):
        sp = [r["speed"] for r in rs]
        worst = min(rs, key=lambda r: r["speed"])
        out.append(f"| {op} | {len(rs)} | {statistics.median(sp):.2f} | {worst['speed']:.2f} "
                   f"({worst['dtype']} {worst['structure']} {worst['view']}) | {sum(s < 0.9 for s in sp)} |")
    out += ["", "## Per dtype (all operations)", "", "| dtype | cells | median speed | below 0.9 |", "|---|---|---|---|"]
    by_dt = collections.defaultdict(list)
    for r in ok:
        by_dt[r["dtype"]].append(r["speed"])
    for dt in A.DTYPES:
        if dt in by_dt:
            sp = by_dt[dt]
            out.append(f"| {dt} | {len(sp)} | {statistics.median(sp):.2f} | {sum(s < 0.9 for s in sp)} |")
    out += ["", "## Slowest 60 cells", "", "| cell | n | PyCauset ms | NumPy ms | speed |", "|---|---|---|---|---|"]
    for r in sorted(ok, key=lambda r: r["speed"])[:60]:
        out.append(f"| {r['key']} | {r['n']} | {r['t_pc'] * 1e3:.2f} | {r['t_np'] * 1e3:.2f} | {r['speed']:.2f} |")
    if overhead:
        out += ["", "## Per-call overhead (4-element operands)", "", "| call | PyCauset µs | NumPy µs |", "|---|---|---|"]
        out += [f"| {k} | {v[0]:.1f} | {v[1]:.1f} |" for k, v in overhead.items()]
    errs = [r for r in rows if "error" in r]
    if errs:
        out += ["", "## Timing errors (fix the harness or the engine)", ""]
        out += [f"* {r['key']}: {r['error']}" for r in errs[:80]]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
