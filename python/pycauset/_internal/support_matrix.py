"""Internal: executable dtype/op support matrix.

Phase 4 goal: make "what is supported" explicit and enforceable.

This file is intentionally conservative about runtime (small shapes), but broad
about *surface area*: it covers core ops across all dtypes, plus representative
mixed-dtype pairs to catch asymmetric dispatch bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SupportCase:
    kind: str  # "matrix" | "vector" | "matvec" | "vecmat" | "vector_scalar"
    op: str
    a_dtype: str
    b_dtype: str | None = None


DTYPES: List[str] = [
    "bit",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex_float16",
    "complex_float32",
    "complex_float64",
]


_MATRIX_OPS_MM_SAME: List[str] = ["add", "sub", "mul", "div", "matmul"]
_MATRIX_OPS_MM_MIXED: List[str] = ["add", "sub", "mul"]
_VECTOR_OPS_VV: List[str] = ["add", "sub", "dot", "outer"]
_MV_OPS: List[str] = ["matvec", "vecmat"]
_UNARY_OPS: List[str] = ["H", "conj"]
_VECTOR_SCALAR_OPS: List[str] = ["add_scalar", "mul_scalar"]

# Scalar tokens are intentionally distinct from DTYPES. They model Python scalar
# inputs to vector operators, not vector storage types.
SCALAR_DTYPES: List[str] = [
    "scalar_int64",
    "scalar_float64",
    "scalar_complex128",
]


def _same_dtype_cases() -> List[SupportCase]:
    cases: List[SupportCase] = []
    for dt in DTYPES:
        for op in _MATRIX_OPS_MM_SAME:
            cases.append(SupportCase("matrix", op, dt, dt))
        for op in _VECTOR_OPS_VV:
            cases.append(SupportCase("vector", op, dt, dt))
        for op in _MV_OPS:
            cases.append(SupportCase(op, op, dt, dt))
        for op in _UNARY_OPS:
            cases.append(SupportCase("matrix", op, dt, None))
            cases.append(SupportCase("vector", op, dt, None))
    return cases


def _mixed_dtype_pairs() -> List[tuple[str, str]]:
    # Representative mixed pairs to catch asymmetric dispatch/promotion.
    # Keep this list short enough to avoid blowing up runtime.
    pairs: List[tuple[str, str]] = [
        ("bit", "int16"),
        ("bit", "int32"),
        ("bit", "int8"),
        ("bit", "uint8"),
        ("bit", "float16"),
        ("bit", "float64"),
        ("int8", "int16"),
        ("int16", "int32"),
        ("int32", "int64"),
        ("uint8", "uint16"),
        ("uint16", "uint32"),
        ("uint32", "int32"),
        ("uint32", "int64"),
        ("int16", "float64"),
        ("int32", "float16"),
        ("float16", "float32"),
        ("float32", "float64"),
        ("complex_float16", "complex_float32"),
        ("complex_float32", "complex_float64"),
        ("float16", "complex_float16"),
        ("float32", "complex_float32"),
        ("float64", "complex_float64"),
        ("float64", "complex_float32"),
        ("complex_float32", "float64"),
    ]

    # Include reverse direction for commutative ops to catch order bugs.
    pairs += [(b, a) for (a, b) in pairs if a != b]
    return pairs


def _mixed_dtype_cases() -> List[SupportCase]:
    cases: List[SupportCase] = []
    for a_dt, b_dt in _mixed_dtype_pairs():
        for op in _MATRIX_OPS_MM_MIXED:
            cases.append(SupportCase("matrix", op, a_dt, b_dt))
        for op in _VECTOR_OPS_VV:
            cases.append(SupportCase("vector", op, a_dt, b_dt))
        # matvec/vecmat mixed: matrix dtype vs vector dtype
        cases.append(SupportCase("matvec", "matvec", a_dt, b_dt))
        cases.append(SupportCase("vecmat", "vecmat", a_dt, b_dt))
    return cases


def _vector_scalar_cases() -> List[SupportCase]:
    cases: List[SupportCase] = []

    # v + s and v * s are supported for int/float scalars across all vector dtypes.
    for v_dt in DTYPES:
        for s_dt in ("scalar_int64", "scalar_float64"):
            cases.append(SupportCase("vector_scalar", "add_scalar", v_dt, s_dt))
            cases.append(SupportCase("vector_scalar", "mul_scalar", v_dt, s_dt))

    # Complex scalar multiplication is only supported for complex vectors.
    for v_dt in DTYPES:
        if not v_dt.startswith("complex_"):
            continue
        cases.append(SupportCase("vector_scalar", "mul_scalar", v_dt, "scalar_complex128"))

    return cases


SUPPORTED: List[SupportCase] = _same_dtype_cases() + _mixed_dtype_cases() + _vector_scalar_cases()


def summarize() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in SUPPORTED:
        key = f"{c.kind}:{c.op}"
        out[key] = out.get(key, 0) + 1
    return out
