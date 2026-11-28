# Out of Core

How PyCauset works with matrices larger than RAM: the storage is memory-mapped and the
kernels run on it; the operating system pages data in and out.

## Placement

Each object is placed once, when it is made ([[internals/Memory and Data|Memory and Data]]): in
RAM when it is below the memory threshold and fits the free RAM (minus a safety margin; the
container's or batch job's cgroup limit counts), otherwise in a memory-mapped temporary file.
Nothing is moved later.

## Execution

Every kernel reads its operands' storage and writes its result's in place, mapped or not.
For file-backed objects the OS reads pages on first touch and drops clean ones under memory
pressure; nothing in the engine copies tiles or reads files explicitly. Results of a call
that does not fit in RAM are placed in files, so element-wise operations, reductions,
matrix-vector products, `eigvals_arnoldi`, matmul and the causal-matrix products work on
data larger than RAM (as fast as the disk and the access pattern allow).

Exception, today: the LAPACK operations (`solve`, `inv`, `determinant`, `cholesky`, `qr`,
`lu`, `svd`, `lstsq`, `eigh`, `eig`) copy the whole operand into a heap work buffer, which is
not checked against free RAM (review 3.5 C-4). On data larger than RAM they need that much
memory.

(Until session 9 a tiled `StreamingManager` ran dense same-dtype matmul and `+ - * /` when
operands and result did not fit in RAM. It was slower than paging in every case measured
and allocated about 1.6x the free RAM, so it is gone; review 3.5.)

## Hints

Callers can declare an access pattern (`pycauset.MemoryHint`, `matrix.hint(h)`); the
block-matrix layer prefetches and drops leaves. The engine's kernels issue no hints.

## Observing it

`pc.last_io_trace(op)` lists each operand's storage (`"memory"`, `"file"`, `"temp"`) and the
kernels that ran. To exercise the out-of-core placement in tests, a huge safety margin makes
the governor see no free RAM (`tests/engine/test_out_of_core.py`).
