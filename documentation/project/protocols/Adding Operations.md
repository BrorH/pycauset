# Protocol: Adding or Changing an Operation

**Status:** authoritative (2026 review, Phase N). Supersedes the device-chain protocol
(`ComputeDevice` → `AutoSolver` → CUDA checklists) and the property-driven dispatch it
described. GPU work is on hold; see `experimental/cuda/README.md`.

**Objective:** an operation that is correct for *every* dtype × structure × view × storage
combination by construction, fast where it matters, and provably so: measured, not hoped.

Why this protocol exists: the code base grew operation by operation, each with branches for
the combinations its author thought of, private copies of the view rules, silent NumPy
fallbacks and result caches. Every unplanned combination was its own unplanned path, so the
same classes of bug kept appearing in new places (review FINDINGS F-064…F-086). The procedure
below makes new combinations correct by construction, and the atlas proves it.

## 1. The shape of every operation

```
Python operator / pycauset.f()        thin binding: no logic, no NumPy, no properties
        │
entry function (one per operation)    src/math/*.cpp   e.g. dispatch_matmul
   1. validate shapes
   2. result dtype = promotion::resolve(op, dtype_a, dtype_b)     (the NumPy rule)
   3. structure rules (declared list, see §3)
   4. normaliser: bring each operand to a kernel contract         (shared tools, §4)
        │
kernel(s) with a written contract     src/compute/cpu/CpuSolver.cpp
   accept exactly what the contract says; throw on anything else
```

Worked example: `dispatch_matmul` (`src/math/LinearAlgebra.cpp`) and `CpuSolver::matmul`
(`src/compute/cpu/CpuSolver.cpp`), review F-080. Five rules and four kernel contracts
replaced ~25 front-end branches and 9 device fallbacks.

## 2. Step by step

### Step 1: write down the semantics (NumPy is the oracle)

- What NumPy returns for the same logical arrays: values, result dtype, and where it raises.
  PyCauset matches NumPy unless a documented PyCauset rule says otherwise (bit operands
  counting in int32, integer matmul raising on overflow); such rules live in
  [[internals/DType System|DType System]].
- The result dtype comes from `promotion::resolve` (`core/PromotionResolver`). If the op
  needs a new rule, add it there, not in the op.
- `bit`: state whether the op is bitwise (stays packed), numeric (widens) or an error.

### Step 2: add it to the atlas first

Add one `Op(...)` line to `OPS` in `tests/atlas/atlas.py` with its NumPy reference. Then:

```
PYTHONPATH=python python -m tests.atlas.report <op>   # every cell of the op
```

The atlas runs the op in every dtype × structure × view × storage cell (and dtype/structure
pairs for binary ops) and classifies each: ✓ fast kernel, ○ generic (per-element virtual
reads), N NumPy round trip, ✗ wrong, — raises where NumPy answers. Before the op exists,
every cell is —: that is the work list.

### Step 3: the entry function

One function per operation (`src/math/`). In order:

1. Shape checks. Raise `std::invalid_argument` with the NumPy-style message.
2. `const DataType r = promotion::resolve(...)`.
3. Structure rules (§3), each a few lines.
4. The normaliser (§4): convert each operand once to what the kernel accepts, then call it.

The entry function does not read properties, does not cache, does not call NumPy, and has no
`try { fast } catch { fallback }`.

### Step 4: the kernels and their contracts

- One kernel per *family*, templated on the element type: `dense_kernel<T>` for every
  float, complex and integer T, not one hand-written loop per dtype pair.
- A comment above each kernel states its contract: which classes and dtypes, which view
  flags it reads natively (typically transposition and a real scale factor), what it
  writes. Anything else is a contract violation: throw, never guess.
- Kernels read storage through typed pointers or the shared readers (§4), never through
  `get_element_as_double/complex` in a loop (the atlas counts those calls as ○ generic).
- Linear algebra (factorisations, eigen problems, det): the entry normalises with
  `to_dense(a, lapack_dtype(...))`; the kernel is one LAPACK template over float, double,
  complex<float>, complex<double> whose contract is "plain dense in the LAPACK dtype"
  (`compute/cpu/CpuLinalg.cpp` is the pattern: `?syevd`/`?heevd`, `?geev`, `?potrf`,
  `?geqrf`, `?getrf`). Single precision stays single precision, as in NumPy.
- Integer arithmetic: element-wise wraps like NumPy; reductions that change magnitude (matmul)
  raise `OverflowError` on overflow of the result dtype (DType System).
- Determinism: parallel reductions sum fixed blocks in a fixed order (see `block_sum` in
  CpuSolver.cpp), so results do not depend on the thread count.

### Step 5: the binding

`src/bindings/bind_*.cpp`: convert arguments, call the entry function, return. No second
entry point for the same operation (one path for `+` and `+=`, F-086); no NumPy; no
property reads; no result caching.

### Step 6: the gates (all must pass before merging)

| Gate | Command | Requirement |
|---|---|---|
| Correctness | `pytest tests/atlas` | 0 ✗ cells, no regression against `review/atlas/atlas.json.gz` |
| Paths | `python -m tests.atlas.report` | ○ generic only where a finding explains it |
| Speed | `python -m tests.atlas.timing <op>` | ≥ 0.9× NumPy per fast cell, or a finding |
| Suites | `pytest tests/engine tests/python` and `PYCAUSET_LAZY=0 pytest tests/engine` | pass |
| Regressions | `tests/engine/test_regressions.py` | one `test_F<id>_...` per bug fixed |

Commit the regenerated `review/atlas/ATLAS.md`, `atlas.json.gz` and `TIMING.md` with the op.

### Step 7: documentation

- API page in `documentation/docs/functions/` or the class page: semantics, result dtype,
  result structure, raises.
- `CHANGELOG.md` for user-visible behaviour.
- If the op needed a new structure rule or kernel family, say so in the entry function's
  comment block (it is the operation's specification).

## 3. Structure rules

A structure rule exists when the *result* has a structure (triangular @ triangular is
triangular) or when the structure changes the algorithm's cost (diagonal @ diagonal is O(n),
a bit matrix stays packed). Each rule:

- covers every dtype the structure's class exists for, not one or two (the old matmul had
  "triangular" for float64 and int32 only);
- states when it applies (e.g. both upper triangular, no window) and otherwise falls
  through to the dense rule;
- states any deviation from NumPy. Skipping structural zeros differs from NumPy only where
  a skipped `0 * inf` would be `nan`; say so.

The structure comes from the operand's **class** (`TriangularMatrix`, `DiagonalMatrix`, …),
never from a property: properties are user metadata and never change a result (F-081).

## 4. Shared tools: use these, never re-derive them

| Need | Tool | Where |
|---|---|---|
| The value of an element through its view | `view_value(raw, factor, conj)` | `include/pycauset/math/ViewValue.hpp` |
| An operand as plain dense of dtype T (every class, every view, typed) | `to_dense(a, dtype)` | `math/ScalarOps.hpp` (uses `read_wide`, `math/ViewRead.hpp`) |
| Same dtype, view folded into storage | `write_out(a)` | `math/ScalarOps.hpp` |
| Logical rows / the diagonal, typed, O(rows) memory (reductions, element-wise, streaming) | `read_rows(a, i0, i1, out)` / `read_diagonal(a, out)` (int64, double or complex128 buffers) | `math/ViewRead.hpp` |
| A structured matrix from another matrix's stored region (same dtype, exact) | `pack_as(a, structure)` | `math/ScalarOps.hpp` |
| NumPy's LAPACK dtype (linalg entries) | `lapack_dtype(a, b)` | `math/LinearAlgebra.hpp` |
| Dtype conversion keeping structure | `cast_to(a, dtype)` | `math/ScalarOps.hpp` |
| Does the operand carry any view? | `is_plain(a)` | `math/ScalarOps.hpp` |
| Typed traversal of any class (one dispatch, then typed `get`) | `visit_matrix(m, f)` / `visit_vector` | `core/DTypeDispatch.hpp` |
| A vector as a 1 × n matrix (one implementation for both) | `as_row_matrix(v)` | `math/VectorOps.hpp` |
| Result dtype | `promotion::resolve(op, a, b)` | `core/PromotionResolver.hpp` |

The view rule is `value = conj?(raw · factor)`, with integer factors wrapping, real storage
times a real factor in real arithmetic, and a +0.0 antisymmetric diagonal. It is written once,
in `ViewValue.hpp`. Four hand-written copies of it existed and two were wrong (F-086): do not
write a fifth.

## 5. Forbidden

- NumPy as a backend (`np.linalg.*` behind a PyCauset call). NumPy is the oracle.
- Silent fallbacks: `try { fast } catch { slow / NumPy }`. Errors surface.
- Reading properties to choose a path or produce a result (they are user notes, F-092);
  caching results outside the planned derived-value cache (F-081, `review/features/DERIVED_CACHE.md`).
- Per-dtype or per-class copy-paste branches where a template over the element type works.
- A private copy of the view rule.
- Guesses deciding results: sampling or heuristics may only produce warnings.
- `get_element_as_*` loops in kernels.

## 6. Performance notes

- An O(n²) conversion in front of an O(n³) kernel is acceptable only if it is typed. A
  virtual-call conversion costs ~100× a BLAS flop and dominates up to n in the thousands
  (F-083).
- For O(n²) operations a conversion doubles the cost: read the common views (transposition,
  real factor) natively in the kernel.
- Parallelise over flat blocks, not rows: a vector is one row.
- Per-call overhead of the Python layers is measured in `TIMING.md` (per-call table).

## 7. Streaming (out-of-core)

Ops that can run tile by tile on file-backed data delegate to `StreamingManager`
(`src/compute/StreamingManager.cpp`), whose tile kernels have the same contract as the dense
kernels (same dtype, dense). See [[internals/Streaming Manager.md|Streaming Manager]].

## Definition of done

- [ ] Semantics written down (NumPy reference; any deviation documented).
- [ ] Atlas entry; `pytest tests/atlas` passes (0 ✗, no regression).
- [ ] Generic cells explained by findings; timing ≥ 0.9× NumPy or a finding.
- [ ] Entry function with structure rules + normaliser; kernels with written contracts.
- [ ] No forbidden pattern (§5).
- [ ] Suites pass in both lazy modes; regression tests for bugs found.
- [ ] API docs and CHANGELOG updated; atlas reports committed.

## See also

- [[project/protocols/Adding Types|Adding Types]]: new structures and dtypes
- [[internals/DType System|DType System]]: promotion, overflow, bit rules
- [[project/protocols/NumPy Alignment Protocol|NumPy Alignment Protocol]]
- `review/features/ATLAS.md`: the atlas design
