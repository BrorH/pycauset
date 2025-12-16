# Warnings & Exceptions

PyCauset is designed for **scale-first** workloads. As a result, we must be explicit when the runtime does something that is:

- surprising (dtype underpromotion, accumulator widening),
- potentially expensive (slow fallback paths), or
- potentially unsafe (overflow risk).

This page defines the project-wide conventions for **user-facing warnings** and **exceptions**.

## When to warn vs. raise

### Use warnings when

Warnings are for cases where execution can continue correctly, but the user should be informed.

- **Policy surprises** (e.g., underpromotion within floats, reduction accumulator widening)
- **Heuristic risk checks** (e.g., integer matmul overflow-risk preflight)
- **Performance hazards** (e.g., a fallback that will be much slower than expected)

Warnings should be:

- **actionable** (tell the user what happened and what they can do),
- **deduplicated** (warn-once policy),
- **filterable** (PyCauset-specific warning categories).

### Raise exceptions when

Exceptions are for correctness and contract failures.

- **Invalid arguments / shape mismatches**
- **Unsupported dtype/structure combinations** (unless there is an explicit, documented fallback)
- **Integer overflow** (policy: no silent wrap, no silent output widening)

Exceptions must be **deterministic** and should include stable, specific messages.

## Warning categories (Python)

PyCauset warnings must use a package-specific category so users can filter them without suppressing unrelated warnings.

Defined in `python/pycauset/_internal/warnings.py` and re-exported from `pycauset`:

- `pycauset.PyCausetWarning` (base)
- `pycauset.PyCausetDTypeWarning` (promotion/accumulator dtype notifications)
- `pycauset.PyCausetOverflowRiskWarning` (heuristic overflow risk preflights)
- `pycauset.PyCausetPerformanceWarning` (slow-path / performance notifications)

### Filtering examples

Users can suppress specific warning families:

- `warnings.filterwarnings("ignore", category=pycauset.PyCausetOverflowRiskWarning)`

## Warning message standard

Warnings should follow this format:

- Start with `pycauset <op> ...` or `pycauset <op> preflight ...`
- Include **operation name** (e.g., `matmul`, `dot`)
- Include **relevant dtypes / structures** (even if the current engine is only int32/float32/float64)
- State what was changed:
  - accumulator dtype,
  - output dtype/storage behavior (explicitly say if it did *not* change)
- State the reason (e.g., “reduction-aware integer width”, “heuristic bound indicates plausible overflow”)
- Provide a mitigation hint when reasonable (e.g., scale inputs, use float output)

Noise control requirements:

- Use a **warn-once** mechanism keyed by a stable identifier (usually `op + dtype tuple`).
- Prefer `stacklevel` so the warning points to user code.

## Where warnings should be emitted

Goal: warnings must fire for the *real user entrypoints*, not only for convenience wrappers.

- Prefer emitting warnings at the **binding funnel** for Python operator calls.
  - Example: `MatrixBase.__matmul__` and `pycauset._pycauset.matmul`.
- Emit warnings in Python wrappers only when the Python wrapper is the canonical entrypoint.
- Avoid emitting warnings deep inside hot kernels.

Rationale:

- The binding funnel has enough context to format user-facing messages.
- The kernel should remain tight; warnings there are hard to dedupe and may hurt performance.

## Exception conventions

### C++ → Python mapping

We rely on pybind11’s standard exception translation and a small amount of explicit translation.

- `std::invalid_argument` → `ValueError` (via `translate_invalid_argument` where used)
- `std::out_of_range` → `IndexError`/`ValueError` depending on binding usage
- `std::overflow_error` → `OverflowError`
- Other `std::runtime_error` → `RuntimeError`

Policy notes:

- **Integer overflow must raise** (no wrap). Prefer `std::overflow_error`.
- Prefer `std::invalid_argument` for shape/contract violations so Python sees `ValueError`.

### Error messages

- Keep messages stable and specific.
- Prefer: `"Integer matmul overflow: ..."` over generic `"overflow"`.

## Current implemented warnings (Phase 1 work)

- `int32 @ int32` matmul accumulator notification (`PyCausetDTypeWarning`)
- mixed `float32/float64` underpromotion notification (`PyCausetDTypeWarning`)
- integer matmul overflow-risk preflight warning (`PyCausetOverflowRiskWarning`)

As dtype expansion progresses, these will generalize to the resolver-selected dtype tuples.
