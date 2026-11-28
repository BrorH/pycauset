# Architecture

The map of the moving parts: what each owns and how a call travels from Python to the
bytes and back. Whoever changes a subsystem's responsibilities or wiring updates this page
in the same commit. Deeper pages: [[internals/Compute Architecture|Compute Architecture]],
[[internals/Memory and Data|Memory and Data]], [[internals/Out of Core|Out of Core]],
[[internals/DType System|DType System]]; the file map is [[internals/Codebase Structure|Codebase Structure]].

## The stack

```
 user code     pc.matrix(...), A @ B, pc.solve(A, b), pc.causet(n=...), pc.save(...)
 ──────────────────────────────────────────────────────────────────────────────────
 Python        __init__.py (public names) -> _internal/ (glue, persistence, block
               matrices); causet.py, spacetime.py, synthetic.py, vis.py on top
 ──────────────────────────────────────────────────────────────────────────────────
 bindings      src/bindings.cpp + src/bindings/*.cpp  ->  pycauset._pycauset
 ──────────────────────────────────────────────────────────────────────────────────
 C++ engine    structure classes        include/pycauset/{matrix,vector}
 (libpycauset  operation entries        src/math (structure rules, normaliser)
  _core)       dispatch and kernels     src/compute: ComputeContext -> CpuSolver
               storage                  PersistentObject, MemoryMapper, MemoryGovernor
               causal sets              src/causet: sprinkler, Minkowski diamond
 ──────────────────────────────────────────────────────────────────────────────────
 libraries     OpenBLAS + LAPACKE, OpenMP, Eigen 3.4 (header-only), Spectra 1.0.1
               (restarted Krylov eigensolvers), pybind11
```

## Object model

Every matrix and vector is a `PersistentObject`: it owns its bytes through a shared
`MemoryMapper` (RAM or a memory-mapped file) and carries **view metadata**: a complex
scale, transposition, conjugation and, for windows, row / column offsets into a larger
storage. Transposing, conjugating, scaling and slicing change only metadata (O(1)).

The class is the structure: `DenseMatrix`, `TriangularMatrix` (strictly upper or with a
diagonal; bit-packed for `bool`: the causal matrix), `SymmetricMatrix`,
`AntiSymmetricMatrix`, `HermitianMatrix`, `SkewHermitianMatrix` (packed upper triangle),
`DiagonalMatrix`, `IdentityMatrix` (no storage), `DenseVector`, `UnitVector`. One Python
class per structure, each for all 16 dtypes (`DataType` and `Structure` in
`core/Types.hpp`). There is no flag or assertion store: what an operation may exploit is
what the class guarantees.

**Views are values.** Storage is shared copy-on-write: the first write to an object whose
storage another object (or a loaded snapshot file) shares gives the writer its own copy
(`ensure_unique`). A write never shows through to another object, unlike NumPy's views.

**One definition of view semantics:** `include/pycauset/math/ViewValue.hpp` (how scale,
transposition and conjugation combine with a stored element). Every reader uses it;
`ViewRead` reads whole blocks of a view into a typed buffer. Underneath, `math/RawRows.hpp`
copies rows of stored values out of dense (windows, transposes), triangular,
symmetric-family and diagonal storage in bulk (no per-element `get`); the view rule is then
applied per element. Whole-matrix kernels (to_dense, element-wise row loads, sums) read
through it.

## Evaluation: eager results, lazy where it pays

Results are computed when the operation runs, except where a lazy form is smaller and
never wrong: `C + C.T` and `C - C.T` of a causal matrix are a `BitClosureMatrix` reading
`C`'s bits (int32 logical dtype, symmetric / antisymmetric structure); matrix-vector
products and eigensolvers use the bits directly, anything else writes the result out first.
`PYCAUSET_LAZY=0` turns this off (the suites run both ways). Design:
`review/features/LAZY.md`.

## An operation, end to end: `A @ B`

1. **Binding** (`bind_matrix.cpp`): one matmul helper for every spelling (`A @ B`,
   `pc.matmul`, NumPy operands, vectors); NumPy arrays are imported first
   (`numpy_import.hpp`).
2. **Entry function** (`src/math`): resolves the result dtype (`promotion::resolve`; a scalar
   operand takes part as an operand of one dtype, DType System §9) and structure
   (`StructureResolver`), normalises the operands (views read in place), and allocates the
   result through `ObjectFactory` (RAM or file: `PersistentObject::initialize_storage`).
3. **Device:** `ComputeContext` holds the `CpuSolver`, which runs the kernel on the
   operands' storage (mapped files out of core: the OS pages; [[internals/Out of Core|Out of Core]]).
4. **Kernel** (`src/compute/cpu`): BLAS for float / complex, exact integer paths, popcount
   for bits, structure-specific loops; it records its tag in the kernel log
   (`DebugTrace`), which `pc.last_io_trace` and the atlas read.
5. **Result:** the Python object of its structure class.

The procedure for adding or changing an operation is
[[project/protocols/Adding Operations|Adding Operations]].

## Storage

Objects up to the memory threshold (default 1 GiB) live in RAM, larger ones in temporary
files in the backing directory; a new object also goes to a file when it would not fit the
`MemoryGovernor`'s RAM budget. Objects are never moved to disk after they are made (kernels
hold raw pointers into their storage, F-120). A temporary file goes with its storage's last
owner (views share the storage). `pc.save` writes a single-file `.pycauset` container
(header slots, payload, typed metadata: [[internals/PyCauset Container Format|Container Format]]);
`pc.load` memory-maps the payload copy-on-write, so the file is never modified.

## Python layer

`__init__.py` defines every public name (thin: argument checks, then the engine) and
adds Python conveniences to the native classes (`repr` / `str` in `_internal/formatting.py`,
factories). `_internal/ops.py` holds the linear-algebra glue, `persistence.py` the
container, `blockmatrix.py` / `thunks.py` the Python-level block matrices
([[internals/Block Matrices|Block Matrices]]). The causal-set layer (`CausalSet`, the
Minkowski diamond, synthetic orders, plots) is built on the engine: a `CausalSet` is a bit
`TriangularMatrix` plus its spacetime and seed.

## Verification

NumPy is the oracle, never a backend. `tests/atlas` enumerates every operation x dtype x
structure x view x storage against NumPy (`review/atlas/ATLAS.md`); `tests/engine` holds
the subject suites and one regression test per fixed bug; the code on every documentation
page runs (`test_doc_snippets.py`).
