# Warnings & Exceptions

PyCauset is designed for **scale-first** workloads. As a result, we must be explicit when the runtime does something that is:

- surprising (a result dtype that differs from NumPy's, e.g. float32/float64 underpromotion),
- potentially expensive (slow fallback paths), or
- potentially unsafe (overflow risk).

This page defines the project-wide conventions for **user-facing warnings** and **exceptions**.

## When to warn vs. raise

### Use warnings when

Warnings are for cases where execution can continue correctly, but the user should be informed.

- **Policy surprises**: a result dtype that differs from NumPy's (float32/float64
  underpromotion, bit + bit → int32). Internal details that don't change the result
  (e.g. which accumulator a reduction uses) are not warned about.
- **Lossy assignments** (e.g., narrowing or float→int casts)
- **Advisory risk estimates** (e.g., the int32 matmul overflow estimate from sampled entries)

Guesses may inform, never decide: a warning may be based on sampling or a heuristic,
but no result (values, dtype, success/failure) may depend on one.
- **Performance hazards** (e.g., a fallback that will be much slower than expected)

Warnings should be:

- **actionable** (tell the user what happened and what they can do),
- **deduplicated by Python's warnings filter** (the default filter shows a warning once
  per call site; `warnings.simplefilter("always")` shows every one),
- **filterable** (PyCauset-specific warning categories).

### Raise exceptions when

Exceptions are for correctness and contract failures.

- **Invalid arguments / shape mismatches**
- **Unsupported dtype/structure combinations** (unless there is an explicit, documented fallback)
- **Integer reduction overflow** (`matmul` final cast doesn't fit the output dtype → `OverflowError`). Elementwise integer arithmetic wraps silently (C/NumPy semantics) and does **not** raise.

Exceptions must be **deterministic** and should include stable, specific messages.

## Warning categories (Python)

PyCauset warnings must use a package-specific category so users can filter them without suppressing unrelated warnings.

Defined in `python/pycauset/_internal/warnings.py` and re-exported from `pycauset`:

- `pycauset.PyCausetWarning` (base)
- `pycauset.PyCausetDTypeWarning` (result dtypes that differ from NumPy's)
- `pycauset.PyCausetOverflowRiskWarning` (narrowing or float→int casts on assignment;
  advisory int32 matmul overflow estimate)
- `pycauset.PyCausetPerformanceWarning` (slow-path / performance notifications)
- `pycauset.PyCausetStorageWarning` (storage/mmap/cache notifications)

### Filtering examples

Users can suppress specific warning families:

- `warnings.filterwarnings("ignore", category=pycauset.PyCausetOverflowRiskWarning)`

## Warning message standard

Warnings should follow this format:

- Start with `pycauset <op> ...`
- Include **operation name** (e.g., `matmul`, `dot`)
- Include **relevant dtypes / structures** (even if the current engine is only int32/float32/float64)
- State what was changed (output dtype/storage behavior)
- State the reason (e.g., “reduction-aware integer width”, “heuristic bound indicates plausible overflow”)
- Provide a mitigation hint when reasonable (e.g., scale inputs, use float output)

Noise control requirements:

- Emit through `bindings_warn::warn_with_category` (C++) or `warnings.warn` (Python); never
  keep a private "already warned" set: it would override the user's warnings filter.
- Pass `stacklevel` so the warning points to user code (Python dedupes per call site).

Slicing/indexing assignment warnings

- When `M[slice] = X` triggers dtype conversion or promotion, emit `PyCausetDTypeWarning` (and `PyCausetOverflowRiskWarning` if the conversion increases overflow risk) while applying the NumPy broadcast/shape rules for the indexed region. Shape-changing assignments must raise.

Suggested message pattern:

- `pycauset invert cache unavailable (missing/corrupt cached object); cached entry ignored`

## Where warnings should be emitted

Goal: warnings must fire for the *real user entrypoints*, not only for convenience wrappers.

- Prefer emitting warnings at the **binding funnel** for Python operator calls.
  - Example: `MatrixBase.__matmul__` and `pycauset._pycauset.matmul`.
- Emit warnings in Python wrappers only when the Python wrapper is the canonical entrypoint.
- Avoid emitting warnings deep inside hot kernels.

Rationale:

- The binding funnel has enough context to format user-facing messages.
- The kernel should remain tight; warnings there have no useful stack level and may hurt performance.

## Exception conventions

### C++ → Python mapping

We rely on pybind11’s standard exception translation and a small amount of explicit translation.

- `std::invalid_argument` → `ValueError` (via `translate_invalid_argument` where used)
- `std::out_of_range` → `IndexError`/`ValueError` depending on binding usage
- `std::overflow_error` → `OverflowError`
- Other `std::runtime_error` → `RuntimeError`

Policy notes:

- **Integer reduction overflow must raise** (e.g. `matmul` final cast). Prefer `std::overflow_error`. (Elementwise integer arithmetic wraps per the documented C/NumPy policy and does not raise.)
- Prefer `std::invalid_argument` for shape/contract violations so Python sees `ValueError`.

### Error messages

- Keep messages stable and specific.
- Prefer: `"Integer matmul overflow: ..."` over generic `"overflow"`.

## Current implemented warnings

- bit + bit / bit @ bit promote to int32 (`PyCausetDTypeWarning`; NumPy keeps bool)
- mixed `float32/float64` underpromotion notification (`PyCausetDTypeWarning`)
- narrowing / float→int cast on assignment (`PyCausetOverflowRiskWarning`)
- int32 @ int32 overflow estimate from sampled entries (`PyCausetOverflowRiskWarning`,
  advisory). Overflow itself is detected exactly (int64 accumulator, `OverflowError`
  when a result doesn't fit), whether or not the advisory fired.

