"""Operation atlas: every operation x dtype x structure x view, checked and classified.

Design: review/features/ATLAS.md. One *cell* is (operation, dtype, structure, view, storage).
For each cell the atlas builds the operand(s) in PyCauset and the same logical values in NumPy,
runs the operation in both, and records

  * outcome: ok | wrong | unsupported (PyCauset raises, NumPy does not) | raises (both raise)
             | unavailable (the operand cannot be built: no such structure for this dtype)
  * path:    fast | written-out | generic | numpy | -  (classified from instrumentation, not
             tagged by hand: kernel trace log, virtual accessor counter, NumPy export counter)
  * the kernels that ran, accessor calls, result dtype vs NumPy's.

Value rules (PyCauset differs from NumPy on purpose in places, review/RESTRUCTURE.md):
  * values are compared using PyCauset's result dtype for the arithmetic (bit + bit counts into
    int32, integer wraparound in the result dtype), at a tolerance set by that dtype;
  * a result *kind* below NumPy's (float result returned as integer, complex as real) is wrong:
    it loses values. A kind above NumPy's (bool -> int32) is a recorded difference, not an error.

Run `python -m tests.atlas.atlas` (from the repo root) to regenerate review/atlas/; the CI test
(tests/atlas/test_atlas.py) runs the correctness part and checks it against KNOWN_GAPS.
"""
from __future__ import annotations

import contextlib
import itertools
import math
import os
import tempfile
import time
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import pycauset as pc
from pycauset._internal import export_guard

N = 16  # matrix side; even (antisymmetric inverses), > the accessor threshold's square root
W = N - 2  # window side

DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float16", "float32", "float64", "complex_float16", "complex_float32", "complex_float64"]
NP_DTYPE = {"bool": np.bool_, "complex_float16": np.complex64, "complex_float32": np.complex64,
            "complex_float64": np.complex128}
KIND_ORDER = {"b": 0, "u": 1, "i": 1, "f": 2, "c": 3}
TOL = {"float16": 4e-3, "complex_float16": 4e-3, "float32": 2e-5, "complex_float32": 2e-5}
LINALG_TOL = {"float16": 5e-2, "complex_float16": 5e-2, "float32": 2e-3, "complex_float32": 2e-3}


def np_dtype(d: str):
    return NP_DTYPE.get(d, getattr(np, d, None))


def kind(d: str) -> str:
    if d.startswith("complex"):
        return "c"
    return np.dtype(np_dtype(d)).kind


def is_int(d: str) -> bool:
    return kind(d) in "iub"


# --------------------------------------------------------------------------- operands

STRUCTURES = ["dense", "triangular", "strict_triangular", "diagonal", "identity", "symmetric",
              "antisymmetric", "causal", "c_plus_ct", "vector", "unit_vector"]


class Unavailable(Exception):
    """The operand cannot be built (structure x dtype not offered)."""


def _values(dtype: str, shape, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dtype == "bool":
        return rng.integers(0, 2, size=shape).astype(bool)
    if is_int(dtype):
        return rng.integers(1, 5, size=shape).astype(np_dtype(dtype))
    re = np.round(rng.uniform(-1, 1, size=shape) * 8) / 8 + 0.0  # exact in float16; no -0.0
    if kind(dtype) == "c":
        im = np.round(rng.uniform(-1, 1, size=shape) * 8) / 8 + 0.0
        return (re + 1j * im).astype(np_dtype(dtype))
    return re.astype(np_dtype(dtype))


def _dominant(a: np.ndarray, dtype: str) -> np.ndarray:
    """Add n on the diagonal: invertible and well conditioned (linear algebra cells)."""
    if dtype == "bool":
        return a | np.eye(a.shape[0], dtype=bool)
    return (a + np.eye(a.shape[0], dtype=a.dtype) * a.dtype.type(a.shape[0])).astype(a.dtype)


def reference(structure: str, dtype: str, seed: int) -> np.ndarray:
    a = _values(dtype, (N, N), seed)
    if structure == "dense":
        return _dominant(a, dtype)
    if structure == "triangular":
        return np.triu(_dominant(a, dtype))
    if structure in ("strict_triangular", "causal"):
        return np.triu(a, 1)
    if structure == "diagonal":
        return np.diag(np.diag(_dominant(a, dtype)))
    if structure == "identity":
        return np.eye(N, dtype=np_dtype(dtype))
    if structure == "symmetric":
        u = np.triu(_dominant(a, dtype))
        return u + np.triu(u, 1).T
    if structure == "antisymmetric":
        u = np.triu(a, 1)
        with np.errstate(over="ignore"):
            # A[j,i] = -A[i,j] as IEEE negation (-0.0 below a stored 0.0, as the engine reads it)
            if dtype == "bool":
                return u ^ u.T
            out = (-u).T.copy()
            iu = np.triu_indices(N)
            out[iu] = u[iu]
            return out
    if structure == "c_plus_ct":
        u = np.triu(a, 1)
        return u | u.T if dtype == "bool" else u + u.T
    if structure == "vector":
        return _values(dtype, (N,), seed)
    if structure == "unit_vector":
        v = np.zeros(N, dtype=np_dtype(dtype))
        v[seed % N] = 1
        return v
    raise ValueError(structure)


def build(structure: str, dtype: str, ref: np.ndarray, seed: int):
    """The PyCauset object holding `ref`, built through the public constructors."""
    try:
        if structure == "dense":
            return pc.matrix(ref, dtype=dtype)
        if structure == "vector":
            return pc.vector(ref, dtype=dtype)
        if structure == "unit_vector":
            if dtype != "float64":
                raise Unavailable("UnitVector is float64 only")
            return pc.UnitVector(N, seed % N)
        if structure == "identity":
            return pc.IdentityMatrix(N, dtype=dtype)
        if structure in ("triangular", "strict_triangular"):
            m = pc.TriangularMatrix(N, dtype=dtype, has_diagonal=structure == "triangular")
            lo = 0 if structure == "triangular" else 1
            for i in range(N):
                for j in range(i + lo, N):
                    if ref[i, j]:
                        m[i, j] = ref[i, j].item()
            return m
        if structure == "causal":
            if dtype != "bool":
                raise Unavailable("causal matrices are bit matrices")
            return pc.causal_matrix(ref)
        if structure == "c_plus_ct":
            if dtype != "bool":
                raise Unavailable("C + C^T is built from a bit causal matrix")
            c = pc.causal_matrix(np.triu(ref, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return c + c.T  # PyCauset counts: int32 values, same as `ref` here for 0/1 data
        cls = {"diagonal": pc.DiagonalMatrix, "symmetric": pc.SymmetricMatrix,
               "antisymmetric": pc.AntiSymmetricMatrix}[structure]
        m = cls(N, dtype=dtype)
        for i in range(N):
            for j in range(i if structure != "antisymmetric" else i + 1, N):
                if structure == "diagonal" and i != j:
                    continue
                if ref[i, j]:
                    m[i, j] = ref[i, j].item()
        return m
    except Unavailable:
        raise
    except Exception as e:  # constructor refuses the dtype
        raise Unavailable(f"{type(e).__name__}: {e}") from None


def _int_factor_ref(a, f):
    with np.errstate(over="ignore"):
        return (a * a.dtype.type(f)).astype(a.dtype) if a.dtype != np.bool_ else a.astype(np.int64) * f


# view name -> (pycauset, numpy, applies-to-vectors)
VIEWS: dict[str, tuple[Callable, Callable, bool]] = {
    "plain": (lambda m: m, lambda a: a, True),
    "T": (lambda m: m.T, lambda a: a.T, True),
    "scaled_int": (lambda m: m * 2, None, True),  # numpy side computed with PyCauset's dtype
    "scaled_float": (lambda m: m * 2.5, None, True),
    "conj": (lambda m: m.conj(), lambda a: np.conj(a) if np.iscomplexobj(a) else a, True),
    "H": (lambda m: m.H, lambda a: (np.conj(a) if np.iscomplexobj(a) else a).T, True),
    "window0": (lambda m: m[0:W, 0:W], lambda a: a[0:W, 0:W], False),
    "window1": (lambda m: m[1:W + 1, 1:W + 1], lambda a: a[1:W + 1, 1:W + 1], False),
    "window_shift": (lambda m: m[0:W, 2:W + 2], lambda a: a[0:W, 2:W + 2], False),
    "vwindow": (lambda m: m[1:W + 1], lambda a: a[1:W + 1], None),  # vectors only
    "lazy_sum": (lambda m: m + m, None, True),
}
LINALG_VIEWS = {"plain", "T", "scaled_int", "scaled_float", "conj", "H", "window0", "window1", "lazy_sum"}


def view_reference(view: str, a: np.ndarray, pc_dtype: str) -> np.ndarray:
    """NumPy values of the view; scaled/lazy views take the dtype PyCauset gives them."""
    if VIEWS[view][1] is not None:
        return VIEWS[view][1](a)
    target = np_dtype(pc_dtype)
    wide = np.complex128 if kind(pc_dtype) == "c" else (np.float64 if kind(pc_dtype) == "f" else object)
    x = a.astype(wide) if wide is not object else a.astype(np.int64).astype(object)
    out = {"scaled_int": lambda: x * 2, "scaled_float": lambda: x * 2.5, "lazy_sum": lambda: x + x}[view]()
    return _cast(out, pc_dtype)


def _cast(x, dtype: str) -> np.ndarray:
    """Cast to dtype with integer wraparound (as PyCauset's integer kernels do)."""
    x = np.asarray(x)
    if dtype == "bool":
        return x.astype(bool)
    if not is_int(dtype):
        return x.astype(np_dtype(dtype))
    info = np.iinfo(np_dtype(dtype))
    span = int(info.max) - int(info.min) + 1

    def wrap(v):
        v = int(np.real(v)) if not isinstance(v, int) else v  # int() truncates toward zero
        return (v - int(info.min)) % span + int(info.min)

    out = np.frompyfunc(wrap, 1, 1)(x.astype(object))
    return np.asarray(out, dtype=np_dtype(dtype)).reshape(x.shape)


# --------------------------------------------------------------------------- operations

@dataclass
class Op:
    name: str
    group: str
    run: Callable  # (x, y) -> result (PyCauset)
    ref: Callable  # (a, b) -> result (NumPy, on arrays in the arithmetic dtype)
    arity: int = 1
    matrix: bool = True
    vector: bool = False
    linalg: bool = False  # floats in the arithmetic, well-conditioned operands, LINALG_VIEWS
    o_n2: bool = True  # O(n^2): a write-out is a cost worth flagging
    exact_dtype: bool = False  # result dtype must equal the resolver's (binary element-wise)
    resolver: str = ""
    compare: str = "values"  # values | sorted | invariant
    ints_ok: bool = True
    bool_ok: bool = True


def _svals(x):
    return np.linalg.svd(np.asarray(x, dtype=np.complex128), compute_uv=False)


OPS: list[Op] = [
    # element-wise, matrix (+) matrix (same dtype, structure, view; a second seed)
    Op("add", "elementwise", lambda x, y: x + y, lambda a, b: a + b, 2, vector=True, exact_dtype=True, resolver="add"),
    Op("sub", "elementwise", lambda x, y: x - y, lambda a, b: a - b, 2, vector=True, exact_dtype=True, resolver="subtract"),
    Op("mul", "elementwise", lambda x, y: x * y, lambda a, b: a * b, 2, vector=True, exact_dtype=True, resolver="elementwise_multiply"),
    Op("div", "elementwise", lambda x, y: x / y, lambda a, b: a / b, 2, vector=True, exact_dtype=True, resolver="divide"),
    Op("neg", "elementwise", lambda x, y: -x, lambda a, b: -a, vector=True, bool_ok=False),
    Op("add_scalar", "elementwise", lambda x, y: x + 3, lambda a, b: a + 3, vector=True),
    Op("radd_scalar", "elementwise", lambda x, y: 3 + x, lambda a, b: 3 + a, vector=True),
    Op("rsub_scalar", "elementwise", lambda x, y: 3 - x, lambda a, b: 3 - a, vector=True),
    Op("mul_scalar_float", "elementwise", lambda x, y: x * 0.5, lambda a, b: a * 0.5, vector=True),
    Op("div_scalar", "elementwise", lambda x, y: x / 2, lambda a, b: a / 2, vector=True),
    Op("rdiv_scalar", "elementwise", lambda x, y: 2 / x, lambda a, b: 2 / a, vector=True),
    Op("mul_scalar_complex", "elementwise", lambda x, y: x * 1j, lambda a, b: a * 1j, vector=True),
    Op("and", "bitwise", lambda x, y: x & y, lambda a, b: a & b, 2, ints_ok=False),
    Op("or", "bitwise", lambda x, y: x | y, lambda a, b: a | b, 2, ints_ok=False),
    Op("xor", "bitwise", lambda x, y: x ^ y, lambda a, b: a ^ b, 2, ints_ok=False),
    Op("invert", "bitwise", lambda x, y: ~x, lambda a, b: ~a, ints_ok=False),
    # reductions
    Op("sum", "reduction", lambda x, y: pc.sum(x), lambda a, b: np.sum(a), vector=True),
    Op("trace", "reduction", lambda x, y: pc.trace(x), lambda a, b: np.trace(a)),
    Op("norm", "reduction", lambda x, y: pc.norm(x), lambda a, b: np.linalg.norm(a), vector=True),
    Op("dot", "reduction", lambda x, y: pc.dot(x, y), lambda a, b: np.dot(a, b), 2, matrix=False, vector=True),
    # products
    Op("matmul", "product", lambda x, y: x @ y, lambda a, b: a @ b, 2, o_n2=False, exact_dtype=True, resolver="matmul"),
    Op("matvec", "product", lambda x, y: x @ VEC[x.dtype], lambda a, b: a @ VEC_REF[DT[0]], 1),
    Op("vecmat", "product", lambda x, y: VEC[x.dtype] @ x, lambda a, b: VEC_REF[DT[0]] @ a, 1),
    Op("outer", "product", lambda x, y: pc.outer(x, y), lambda a, b: np.outer(a, b), 2, matrix=False, vector=True),
    # linear algebra
    Op("inv", "linalg", lambda x, y: pc.invert(x), lambda a, b: np.linalg.inv(a), linalg=True, o_n2=False),
    Op("solve", "linalg", lambda x, y: pc.solve(x, RHS[0]), lambda a, b: np.linalg.solve(a, RHS_REF[0]), linalg=True, o_n2=False),
    Op("det", "linalg", lambda x, y: pc.determinant(x), lambda a, b: np.linalg.det(a), linalg=True, o_n2=False),
    Op("slogdet", "linalg", lambda x, y: pc.slogdet(x), lambda a, b: np.linalg.slogdet(a), linalg=True, o_n2=False, compare="slogdet"),
    Op("eigvals", "linalg", lambda x, y: pc.eigvals(x), lambda a, b: np.linalg.eigvals(a), linalg=True, o_n2=False, compare="sorted"),
    Op("eigvalsh", "linalg", lambda x, y: pc.eigvalsh(x), lambda a, b: np.linalg.eigvalsh(a), linalg=True, o_n2=False, compare="sorted"),
    Op("svdvals", "linalg", lambda x, y: pc.svdvals(x), lambda a, b: _svals(a), linalg=True, o_n2=False, compare="sorted"),
    Op("qr", "linalg", lambda x, y: pc.qr(x), lambda a, b: None, linalg=True, o_n2=False, compare="qr"),
    Op("lu", "linalg", lambda x, y: pc.lu(x), lambda a, b: None, linalg=True, o_n2=False, compare="lu"),
    Op("cholesky", "linalg", lambda x, y: pc.cholesky(x), lambda a, b: np.linalg.cholesky(a), linalg=True, o_n2=False, compare="cholesky"),
    Op("pinv", "linalg", lambda x, y: pc.pinv(x), lambda a, b: np.linalg.pinv(a), linalg=True, o_n2=False),
    Op("matrix_power", "linalg", lambda x, y: pc.matrix_power(x, 3), lambda a, b: np.linalg.matrix_power(a, 3), o_n2=False),
    Op("lstsq", "linalg", lambda x, y: pc.lstsq(x, RHS[0]), lambda a, b: np.linalg.lstsq(a, RHS_REF[0], rcond=None)[0], linalg=True, o_n2=False, compare="first"),
    # construction, conversion, indexing, persistence
    Op("export", "io", lambda x, y: x, lambda a, b: a, vector=True),
    Op("getitem", "io", lambda x, y: [x[i, (i * 5) % x.shape[1]] for i in range(x.shape[0])],
       lambda a, b: [a[i, (i * 5) % a.shape[1]] for i in range(a.shape[0])]),
    Op("save_load", "io", lambda x, y: _save_load(x), lambda a, b: a, vector=True),
]
OP_BY_NAME = {o.name: o for o in OPS}

# per-cell operands for matvec/solve (set by run_cell; the lambdas above read them)
VEC: dict = {}
VEC_REF: dict = {}
DT: list = [""]
RHS: list = [None]
RHS_REF: list = [None]


def _save_load(x):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.pycauset")
        pc.save(x, p)
        y = pc.load(p)
        out = _to_np(y)
        with contextlib.suppress(Exception):
            y.close()
        return out


# --------------------------------------------------------------------------- one cell

@dataclass
class Cell:
    op: str
    dtype: str
    structure: str
    view: str
    storage: str = "ram"
    dtype2: str = ""  # second operand dtype (dtype-pair cells)
    structure2: str = ""  # second operand structure (structure-pair cells)
    outcome: str = ""
    path: str = ""
    kernels: list = field(default_factory=list)
    accesses: int = 0
    exports: int = 0
    result_dtype: str = ""
    numpy_dtype: str = ""
    error: str = ""
    seconds: float = 0.0

    def key(self) -> str:
        parts = [self.op, self.dtype + (f"x{self.dtype2}" if self.dtype2 else ""),
                 self.structure + (f"x{self.structure2}" if self.structure2 else ""), self.view, self.storage]
        return "/".join(parts)


def _to_np(r):
    """Export a result without counting it as a NumPy fallback of the operation."""
    if isinstance(r, (np.ndarray, np.generic, bool, int, float, complex)):
        return np.asarray(r)
    out = np.asarray(pc.to_numpy(r, allow_huge=True))
    export_guard.export_count -= 1
    return out


def _pc_dtype_of(r) -> str:
    d = getattr(r, "dtype", None)
    if d is not None and not isinstance(r, np.ndarray):
        return str(d)
    if isinstance(r, bool):
        return "bool"
    if isinstance(r, int):
        return "int64"
    if isinstance(r, float):
        return "float64"
    if isinstance(r, complex):
        return "complex_float64"
    if isinstance(r, np.ndarray):
        return {"complex64": "complex_float32", "complex128": "complex_float64"}.get(str(r.dtype), str(r.dtype))
    return type(r).__name__


def _close(got: np.ndarray, ref: np.ndarray, dtype: str, linalg: bool) -> bool:
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape and got.size == ref.size:
        got = got.reshape(ref.shape)
    if got.shape != ref.shape:
        return False
    if is_int(dtype) and not linalg:
        return np.array_equal(got.astype(object), ref.astype(object)) if dtype != "bool" else np.array_equal(got.astype(bool), ref.astype(bool))
    tol = (LINALG_TOL if linalg else TOL).get(dtype, 1e-8 if linalg else 1e-12)
    g = got.astype(np.complex128)
    r = ref.astype(np.complex128)
    if dtype in ("float16", "complex_float16"):  # the result dtype overflows at 65504 (inf), as NumPy's
        # Entries near the overflow threshold have no well-defined float16 answer (intermediate
        # rounding decides whether they overflow): skip them.
        near = np.maximum(np.abs(r.real), np.abs(r.imag)) > 32752
        g, r = np.where(near, 0, g), np.where(near, 0, r)
    special = ~np.isfinite(r) | ~np.isfinite(g)
    if special.any():
        same = (np.isnan(g) & np.isnan(r)) | (g == r)
        if not np.all(same[special]):
            return False
        g, r = np.where(special, 0, g), np.where(special, 0, r)
    scale = max(1.0, float(np.max(np.abs(r))) if r.size else 1.0)
    return bool(np.all(np.abs(g - r) <= tol * scale + tol * np.abs(r)) or np.allclose(g, r, rtol=tol, atol=tol * scale, equal_nan=True))


def _wide(dtype: str):
    k = kind(dtype)
    return np.complex128 if k == "c" else np.float64


def _arith_inputs(arrays, dtype: str, linalg: bool):
    """Reference inputs: exact Python ints for integer results, else wide floats/complex.

    Complex inputs stay complex even when the result is real (norm, abs)."""
    if any(np.iscomplexobj(x) for x in arrays):
        return [np.asarray(x).astype(np.complex128) for x in arrays]
    if linalg or not is_int(dtype):
        return [np.asarray(x).astype(np.complex128 if kind(dtype) == "c" else np.float64) for x in arrays]
    if dtype == "bool":
        return [np.asarray(x).astype(bool) for x in arrays]
    return [np.asarray(x).astype(np.int64).astype(object) for x in arrays]


def _match_spectra(got, ref, dtype: str) -> bool:
    """Same multiset of (complex) values up to tolerance, by greedy nearest matching."""
    g = list(np.asarray(got).astype(np.complex128).ravel())
    r = np.asarray(ref).astype(np.complex128).ravel()
    if len(g) != len(r):
        return False
    tol = LINALG_TOL.get(dtype, 1e-8) * max(1.0, float(np.max(np.abs(r))) if r.size else 1.0)
    for v in r:
        k = int(np.argmin([abs(v - w) for w in g]))
        if abs(v - g[k]) > tol:
            return False
        g.pop(k)
    return True


def run_cell(cell: Cell) -> Cell:
    op = OP_BY_NAME[cell.op]
    # Stable across processes (str hash() is randomised per process: runs were not reproducible).
    seed = zlib.crc32("/".join((cell.op, cell.dtype, cell.structure, cell.view)).encode()) % 1000
    view_fn = VIEWS[cell.view][0]
    threshold = pc.get_memory_threshold()
    try:
        if cell.storage == "file":
            pc.set_memory_threshold(1024)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return _run_cell(cell, op, seed, view_fn)
            except Exception as e:  # a bug in the atlas itself: visible, never silent
                cell.outcome, cell.error = "harness", f"{type(e).__name__}: {str(e)[:200]}"
                return cell
    finally:
        if cell.storage == "file":
            pc.set_memory_threshold(threshold if threshold is not None else 8 * 1024**3)


def _operand(structure, dtype, view, seed, linalg):
    for attempt in range(25):
        a = reference(structure, dtype, seed + 1000 * attempt)
        m = build(structure, dtype, a, seed)
        v = VIEWS[view][0](m)
        vdt = _pc_dtype_of(v)
        x = view_reference(view, a, vdt)
        # Linear algebra on a numerically singular input has no well-defined answer (det,
        # slogdet, inv are rounding noise there): draw again. Random 0/1 matrices often are.
        if not linalg or x.ndim != 2 or np.linalg.cond(x.astype(np.complex128)) < 1e8:
            return v, x, vdt
    return v, x, vdt


def _run_cell(cell: Cell, op: Op, seed: int, view_fn) -> Cell:
    s2 = cell.structure2 or cell.structure
    d2 = cell.dtype2 or cell.dtype
    try:
        x, a, xdt = _operand(cell.structure, cell.dtype, cell.view, seed, op.linalg)
        y, b, ydt = (_operand(s2, d2, cell.view, seed + 1, op.linalg) if op.arity == 2 else (None, None, ""))
    except Unavailable as e:
        cell.outcome, cell.error = "unavailable", str(e)[:200]
        return cell
    except Exception as e:  # a view PyCauset does not offer (e.g. slicing a structure)
        cell.outcome, cell.error = "unavailable", f"view: {type(e).__name__}: {str(e)[:160]}"
        return cell

    # side operands for matvec / solve, in the operand's dtype
    DT[0] = xdt
    vref = _values(xdt if xdt in DTYPES else "float64", (a.shape[-1],), seed + 7)
    try:
        VEC[x.dtype] = pc.vector(vref, dtype=xdt) if xdt in DTYPES else None
    except Exception:
        VEC[x.dtype] = None
    VEC_REF[xdt] = vref
    rhs = np.linspace(-1, 1, a.shape[0])
    RHS[0], RHS_REF[0] = pc.vector(rhs), rhs

    # PyCauset
    export_guard_before = export_guard.export_count
    pc._native._debug_kernel_log_start()
    pc._native._debug_access_count_start()
    t0 = time.perf_counter()
    err = None
    try:
        r = op.run(x, y)
        # A lazy closure's work happens when it is evaluated: evaluate it inside the window.
        # Exporting the result to NumPy is the `export` operation's cost, not this one's.
        if isinstance(r, pc._native.MatrixBase) and pc._native._debug_is_closure(r):
            r = pc._native.materialize(r)
        if op.name == "export":  # the export is this operation
            got_dtype, got = _pc_dtype_of(r), _materialize(r, op)
    except Exception as e:
        err = e
    cell.seconds = time.perf_counter() - t0
    cell.accesses = int(pc._native._debug_access_count_stop())
    cell.kernels = list(dict.fromkeys(pc._native._debug_kernel_log_stop()))
    cell.exports = export_guard.export_count - export_guard_before
    if err is None and op.name != "export":
        try:
            got_dtype = _pc_dtype_of(r)
            got = _materialize(r, op)
        except Exception as e:
            err = e

    # NumPy natural dtype (for the kind rule) and the reference values in PyCauset's dtype
    np_err = None
    with np.errstate(all="ignore"):
        try:
            nat = op.ref(*( [a] if op.arity == 1 else [a, b]))
            if isinstance(nat, tuple):
                nat = nat[1] if op.compare == "slogdet" else nat[0]
            cell.numpy_dtype = str(np.asarray(nat).dtype) if nat is not None else ""
        except Exception as e:
            np_err = e

    if err is not None:
        cell.error = f"{type(err).__name__}: {str(err)[:200]}"
        cell.outcome = "raises" if np_err is not None or _numpy_rejects(op, cell) else "unsupported"
        cell.path = "-"
        return cell
    cell.result_dtype = got_dtype
    arith = got_dtype if got_dtype in DTYPES else xdt
    try:
        ok = _check(op, cell, got, a, b, arith, xdt, ydt)
    except Exception as e:
        ok = False
        cell.error = f"check: {type(e).__name__}: {str(e)[:160]}"
    if ok and op.exact_dtype and not cell.structure2:
        try:
            promised = pc._debug_resolve_promotion(op.resolver, xdt, ydt)["result_dtype"]
            if promised != got_dtype and promised.replace("bit", "bool") != got_dtype:
                ok, cell.error = False, f"dtype {got_dtype}, resolver promised {promised}"
        except Exception:
            pass
    if ok and cell.numpy_dtype and got_dtype in DTYPES:
        nk = np.dtype(cell.numpy_dtype).kind
        if KIND_ORDER.get(kind(got_dtype), 0) < KIND_ORDER.get(nk, 0) and not (nk == "f" and op.group == "reduction" and op.name == "norm"):
            ok, cell.error = False, f"result kind {got_dtype} below NumPy's {cell.numpy_dtype}"
    cell.outcome = "ok" if ok else "wrong"
    cell.path = classify(cell, op)
    return cell


def _numpy_rejects(op: Op, cell: Cell) -> bool:
    """Cells NumPy itself has no meaning for: raising is the right answer."""
    if op.group == "bitwise" and kind(cell.dtype) in "fc":
        return True
    if op.name in ("eigvalsh", "cholesky") and cell.structure not in ("symmetric", "diagonal", "identity"):
        return True  # not Hermitian/positive definite: no defined answer
    return False


def _materialize(r, op: Op):
    if isinstance(r, tuple):
        return tuple(_to_np(t) if not isinstance(t, (int, float, complex)) else t for t in r)
    if isinstance(r, list):
        return np.asarray(r)
    return _to_np(r)


def _check(op: Op, cell: Cell, got, a, b, arith: str, xdt: str, ydt: str) -> bool:
    linalg = op.linalg or kind(arith) in "fc" and op.group == "linalg"
    if op.compare == "qr":
        q, r_ = got
        return _close(q @ r_, a.astype(np.complex128), arith, True)
    if op.compare == "lu":
        p, l, u = got
        recon = np.asarray(p) @ np.asarray(l) @ np.asarray(u)
        return _close(recon, a.astype(np.complex128), arith, True) or _close(np.asarray(p).T @ np.asarray(l) @ np.asarray(u), a, arith, True)
    if op.compare == "cholesky":
        l = np.asarray(got).astype(np.complex128)
        return _close(l @ l.conj().T, a.astype(np.complex128), arith, True)
    if op.compare == "slogdet":
        s, ld = got
        rs, rld = np.linalg.slogdet(a.astype(np.complex128 if np.iscomplexobj(a) else np.float64))
        return _close(np.asarray(ld), np.asarray(rld), "float64", True) and _close(np.asarray(s), np.asarray(rs), "float64", True)
    if op.compare == "first":
        got = got[0] if isinstance(got, tuple) else got
    inputs = _arith_inputs([a] if op.arity == 1 or b is None else [a, b], arith if not op.linalg else xdt, op.linalg)
    if op.name in ("matvec", "vecmat"):
        inputs = [inputs[0]]
    with np.errstate(all="ignore"):
        ref = op.ref(*(inputs + [None] * (2 - len(inputs))))
    if isinstance(ref, tuple):
        ref = ref[0]
    ref = np.asarray(ref)
    if op.compare == "sorted":
        return _match_spectra(got, ref, xdt if xdt in DTYPES else arith)
    if is_int(arith) and not op.linalg and op.group != "linalg":
        ref = _cast(ref, arith)
    elif op.group == "linalg" and is_int(arith):
        ref = _cast(ref, arith)
    return _close(got, ref, arith, op.linalg)


def classify(cell: Cell, op: Op) -> str:
    if cell.exports > 0:
        return "numpy"
    # Per-element accessor use is the generic path, also for O(n^3) operations: a virtual-call
    # conversion costs ~100x a BLAS flop and dominates the kernel up to n in the thousands
    # (measured by the timing pass, tests/atlas/timing.py). The threshold is one call per
    # element of a row (N), not a fixed count: a 16-element vector op makes 32 calls, which a
    # fixed threshold of 64 classified as fast.
    if cell.accesses >= N:
        return "generic"
    if "write_out" in cell.kernels:
        return "written-out" if op.o_n2 else "fast"
    return "fast"


# --------------------------------------------------------------------------- enumeration

def applicable(op: Op, dtype: str, structure: str, view: str) -> bool:
    is_vec = structure in ("vector", "unit_vector")
    if is_vec and not op.vector:
        return False
    if not is_vec and not op.matrix:
        return False
    if view == "vwindow" and not is_vec:
        return False
    if is_vec and VIEWS[view][2] is False:
        return False
    if op.linalg and view not in LINALG_VIEWS:
        return False
    if op.name == "dot" and view not in ("plain", "conj", "T"):
        return False
    if op.name in ("eigvalsh", "cholesky") and structure not in ("symmetric", "diagonal", "identity"):
        return False  # defined for Hermitian input only (NumPy reads one triangle and trusts it)
    if op.name == "cholesky" and view == "lazy_sum" and dtype == "bool":
        return False
    return True


def enumerate_cells(storages=("ram",)) -> list[Cell]:
    cells = []
    for op in OPS:
        for structure in STRUCTURES:
            for dtype in DTYPES:
                for view in VIEWS:
                    if applicable(op, dtype, structure, view):
                        for st in storages:
                            if st == "file" and view != "plain":
                                continue
                            cells.append(Cell(op.name, dtype, structure, view, st))
    # every dtype pair, dense/plain, for the binary element-wise ops and matmul
    for name in ("add", "sub", "mul", "div", "matmul"):
        for d1, d2 in itertools.product(DTYPES, DTYPES):
            if d1 != d2:
                cells.append(Cell(name, d1, "dense", "plain", dtype2=d2))
    # every structure pair, float64/plain
    mats = [s for s in STRUCTURES if s not in ("vector", "unit_vector", "causal", "c_plus_ct")]
    for name in ("add", "sub", "mul", "matmul"):
        for s1, s2 in itertools.product(mats, mats):
            if s1 != s2:
                cells.append(Cell(name, "float64", s1, "plain", structure2=s2))
    return cells


SYMBOL = {("ok", "fast"): "✓", ("ok", "written-out"): "◐", ("ok", "generic"): "○", ("ok", "numpy"): "N",
          "wrong": "✗", "unsupported": "—", "raises": "·", "unavailable": " "}


def symbol(c: Cell) -> str:
    if c.outcome == "ok":
        return SYMBOL[("ok", c.path)]
    return SYMBOL[c.outcome]


if __name__ == "__main__":  # pragma: no cover - report generation
    from tests.atlas.report import main

    main()
