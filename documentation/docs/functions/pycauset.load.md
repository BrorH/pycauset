# pycauset.load

```python
pycauset.load(path: str)
```

Loads a matrix, vector, block matrix, or `CausalSet` from a binary file created by PyCauset.

## Parameters

*   **path** (*str*): The path to the file to load.

## Returns

*   **Matrix/vector/BlockMatrix/CausalSet**: The appropriate object for the container’s contents.

	For `matrix_type = "BLOCK"`, this returns a `BlockMatrix` reconstructed from a manifest stored in the container file and child blocks stored in a sibling `.blocks/` directory.

## Description

`pycauset.load()` opens a `.pycauset` file and returns an object backed by the file’s memory-mapped payload.

Snapshot semantics:

- A `.pycauset` file is treated as an immutable snapshot.
- Mutating an object loaded from disk does not implicitly overwrite the on-disk snapshot.
- Persisting payload changes requires an explicit save.

Caching:

- Cached-derived values are restored from typed metadata when valid.

## Examples

```python
import pycauset as pc

A = pc.matrix(((1.0, 2.0), (3.0, 4.0)))
pc.save(A, "A.pycauset")
A2 = pc.load("A.pycauset")
assert A2.shape == (2, 2)

# Block matrix load
blk = pc.matrix(((1.0, 0.0), (0.0, 1.0)))
BM = pc.matrix(((blk, blk), (blk, blk)))
pc.save(BM, "bm.pycauset")
BM2 = pc.load("bm.pycauset")
assert BM2.shape == (4, 4)
```

## Block matrices (sidecar layout)

If the container is a block matrix (`matrix_type = "BLOCK"`), load expects:

- `bm.pycauset` (container file)
- `bm.pycauset.blocks/` (sidecar directory)
	- `block_r{r}_c{c}.pycauset` children

The block manifest pins each child’s `payload_uuid`; load validates these pins and fails deterministically on mismatch. Missing sidecar entries or mismatched child files error rather than silently mixing snapshots. View blocks are loaded from the materialized child files (no view references on disk in Release 1).

See [[internals/Block Matrices.md|Block Matrices]] for details.

## See Also

*   `pycauset.CausalSet.load`: For loading `CausalSet` objects from `.pycauset` containers (see [[docs/classes/spacetime/pycauset.CausalSet.md|pycauset.CausalSet]]).

*   [[guides/Storage and Memory]]: Overview of persistence, caching, and mutation semantics.

*   [[internals/Block Matrices.md|Block Matrices]]

