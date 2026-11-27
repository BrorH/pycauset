# pycauset.save

```python
pycauset.save(obj, path: str)
```

Saves a matrix, vector or block matrix to a permanent location on disk.

This writes a new `.pycauset` **snapshot container**.

*   **For native matrices/vectors**: writes a new container file by copying the payload data (via the object’s `copy_storage(...)` implementation) and recording metadata.
*   **For block matrices**: writes a small container file plus a sibling sidecar directory holding the child blocks (see below). Thunk blocks are evaluated **blockwise** (no global densify). Saving raises deterministically if any captured input is stale.

!!! note "Snapshot semantics"
	`.pycauset` files are treated as immutable snapshots. Mutating a loaded object does not implicitly overwrite the on-disk snapshot.

## Parameters

*   **obj** (*matrix, vector, or BlockMatrix*): The object to save.
*   **path** (*str*): The destination path.

## Example

```python
import pycauset as pc

pc.save(pc.matrix([[1.0, 2.0], [3.0, 4.0]]), "data.pycauset")

# Save a block matrix (creates a sidecar directory)
A = pc.matrix(((1.0, 0.0), (0.0, 1.0)))
BM = pc.matrix(((A, A), (A, A)))
pc.save(BM, "bm.pycauset")  # writes bm.pycauset and bm.pycauset.blocks/
```

## See Also


*   [[guides/Storage and Memory]]: Persistence overview, including cache persistence.

## Block matrices (sidecar layout)

When saving a `BlockMatrix`, PyCauset writes:

- Container file: `bm.pycauset`
- Sidecar directory: `bm.pycauset.blocks/`
	- Child files: `block_r{r}_c{c}.pycauset`

The container stores a `block_manifest` that records partitions and references child blocks. Manifest entries pin each child’s `payload_uuid`; load validates the pins.

Additional policies:

- `SubmatrixView` blocks are materialized **block-locally** to stable child files (no multi-block densify).
- Overwrite cleanup deletes only deterministic child filenames inside the sidecar.
- Saves stage child files (and nested sidecars) before commit to reduce partial-update risk.

See [[internals/Block Matrices.md|Block Matrices]] for details.

