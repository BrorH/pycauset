# R1_STORAGE — Single-File Persistence Container + Typed Metadata (Release 1)

**Status:** Implemented for Release 1 (plan + implementation aligned)

**Last updated:** 2025-12-21

> Documentation note:
>
> This file is a planning/spec artifact. User-visible storage behavior and the R1 container format are documented in:
>
> - `documentation/guides/Storage and Memory.md` (canonical: snapshots, mutation, caches, and on-disk format)

## Implementation status (as of this date)

This plan’s frozen “Format summary” is implemented in the Python persistence layer and covered by storage tests.

- Implementation: `python/pycauset/_internal/persistence.py`
- Key tests:
  - `tests/python/test_storage_hard_break.py`
  - `tests/python/test_storage_crash_consistency.py`
  - `tests/python/test_storage_debug_tool.py`

## Purpose

Release 1 needs a **single-file `.pycauset` container format** that:

- is memory-mappable for large payloads,
- supports tiered storage and out-of-core workflows,
- stores **sparse, typed, forward-compatible metadata** (including `properties` from R1_PROPERTIES),
- and allows metadata updates without shifting the payload.

This plan is intentionally *about storage mechanics*. The semantics of `properties` (gospel claims, propagation, etc.) are defined in:

- `documentation/internals/plans/completed/R1_PROPERTIES_PLAN.md`

The key contract between the two plans is:

- the C++/Python frontends continue to call the same high-level save/load APIs;
- only the on-disk representation and the internal storage plumbing changes.

## Non-negotiable constraints

- **No data scans:** persistence code must not require scanning payload to validate metadata.
- **Payload must remain mmap-friendly:** large numeric payloads must be accessible via stable offsets.
- **Sparse metadata:** missing keys remain missing (unset/default) to preserve tri-state semantics.
- **Forward compatibility:** older readers can ignore unknown metadata keys safely.
- **Deterministic layout rules:** the same content + metadata must produce deterministic decisions (even if bytes differ due to appended metadata).

## Scope (what this plan does and does not decide)

This plan specifies the persistence container mechanics and typed metadata encoding.

In scope:

- A single-file container with stable payload offsets (mmap-friendly).
- A typed, sparse metadata representation that is forward-compatible.
- Unambiguous encoding of the metadata taxonomy (identity/header vs view-state vs `properties` + cached-derived).
- A crash-safe metadata update mechanism.

Out of scope for R1_STORAGE (must not silently creep in):

- Multiple independent objects per `.pycauset` file (one file = one object).
- Transparent compression of the payload region (payload must remain directly mappable).
- “Database features” (transactions across multiple files, indexing, etc.).

## Current state (baseline)

There is exactly **one** on-disk format for `.pycauset`: the single-file binary container specified below.

## File format sketch (Release 1 direction)

## Format summary (frozen for R1; implement exactly)

This section is the **Phase 0 contract freeze**. It removes ambiguity by specifying exact binary layouts and encoding rules.

### Endianness

- R1 files are **little-endian only**.
- The header includes an endian marker so readers can fail fast and deterministically if opened on an incompatible platform.

### Alignment

- `payload_offset` MUST be aligned to **4096 bytes** (minimum). (Implementations may choose a larger alignment, but it must be a power-of-two multiple of 4096.)
- `metadata_offset` MUST be aligned to **16 bytes**.

### Fixed header region

The file begins with a fixed-size header region of **4096 bytes**.

- It contains:
  - a file preamble, and
  - two header slots (A and B) used for crash-safe pointer updates.

All integer fields are unsigned little-endian unless specified.

#### File preamble layout (offset 0)

| Field | Type | Notes |
|---|---:|---|
| `magic` | 8 bytes | ASCII `PYCAUSET` |
| `format_version` | u32 | R1 = 1 |
| `endian` | u8 | 1 = little-endian |
| `header_bytes` | u16 | R1 = 4096 |
| `reserved0` | u8[1] | must be 0 |

Immediately following the preamble are two fixed-size slots.

#### Header slot layout (A and B)

Each slot is **128 bytes** and appears twice:

- preamble is exactly 16 bytes; slot A begins at offset 16
- slot B begins at offset 16 + 128

Slot layout:

| Field | Type | Notes |
|---|---:|---|
| `generation` | u64 | monotonic counter; higher wins |
| `payload_offset` | u64 | aligned to 4096 |
| `payload_length` | u64 | bytes |
| `metadata_offset` | u64 | aligned to 16 |
| `metadata_length` | u64 | bytes |
| `hot_offset` | u64 | 0 in R1 unless implemented |
| `hot_length` | u64 | 0 in R1 unless implemented |
| `slot_crc32` | u32 | CRC32 of the first 7 fields (56 bytes) |
| `slot_reserved` | u8[68] | must be 0 (future expansion) |

Validity rules:

- A slot is valid iff:
  - `slot_crc32` matches, AND
  - `payload_offset/payload_length/metadata_offset/metadata_length` are in-range for the file size, AND
  - required alignments are satisfied.
- The active slot is the valid slot with the highest `generation`.
- If neither slot is valid, loading fails.

Crash-consistent update rule:

1) Write the new metadata block at the end of the file.
2) Ensure it is fully written (and flushed if the implementation uses explicit flush).
3) Write the *inactive* header slot with `generation = active.generation + 1` and the new metadata pointer.
4) (Optional but recommended) Flush the header region.

This guarantees $O(1)$ load (choose slot; validate pointer) with no scanning.

### Payload region

- The payload is a raw backing store identical to what current native objects can mmap.
- The payload begins at `payload_offset` and spans `payload_length` bytes.
- Payload interpretation is defined by **identity/header metadata** plus a **payload layout descriptor** (see below).

### Metadata blocks (append-only)

Metadata is stored as one or more blocks appended after the payload. The header slot points at the authoritative block.

#### Metadata block framing (at `metadata_offset`)

| Field | Type | Notes |
|---|---:|---|
| `block_magic` | 4 bytes | ASCII `PCMB` |
| `block_version` | u32 | R1 = 1 |
| `encoding_version` | u32 | typed-metadata encoding version; R1 = 1 |
| `reserved0` | u32 | must be 0 |
| `payload_length` | u64 | bytes of encoded metadata payload |
| `payload_crc32` | u32 | CRC32 of encoded metadata payload |
| `reserved1` | u32 | must be 0 |
| `payload` | bytes | length = `payload_length` |

Validity rules:

- If the framing fields are malformed or `payload_crc32` fails, loading fails deterministically.
- Readers must reject unknown `block_version` or `encoding_version` (clear error).

### Typed metadata encoding v1 (R1 = encoding_version 1)

Encoded metadata payload represents a single **top-level map**.

#### Limits (safety; deterministic failure)

- Max recursion depth: 32
- Max map entries: 1,000,000 (practical cap; R1 typical is tiny)
- Max string length: 16 MiB
- Max bytes length: 1 GiB (for very large blob references; prefer external blocks)

#### Value tags

Each value is encoded as a 1-byte tag followed by a tag-specific payload:

| Tag | Meaning | Encoding |
|---:|---|---|
| 0x01 | Bool | u8 (0/1) |
| 0x02 | I64 | i64 |
| 0x03 | U64 | u64 |
| 0x04 | F64 | f64 |
| 0x05 | String | u32 byte_len + UTF-8 bytes |
| 0x06 | Bytes | u32 byte_len + bytes |
| 0x07 | Array | u32 count + `count` values (each value is tag+payload) |
| 0x08 | Map | u32 count + `count` key/value pairs |

Map encoding:

- `Map` value payload is:
  - u32 count
  - repeated `count` times:
    - key: u16 key_len + UTF-8 bytes
    - value: encoded value (tag + payload)

Notes:

- This encoding is sparse by construction: absent keys are absent.
- Forward compatibility: unknown keys and even unknown nested maps must be skippable by type/length framing.
- Numeric width/sign: R1 standardizes on `I64`/`U64`/`F64`. If smaller widths are needed in later releases, they are added as new tags without breaking R1 readers.

### Required metadata keys (R1 minimum)

The top-level map MUST contain (at minimum) enough identity/header metadata to interpret the payload:

- `rows`: U64
- `cols`: U64
- `matrix_type`: String (stable name)
- `data_type`: String (stable name)
- `payload_layout`: Map (payload layout descriptor)

`payload_layout` (descriptor) must be a small Map. R1 minimum:

- `kind`: String (e.g., `raw_dense`, `raw_triangular`, `raw_bitpacked`)
- `params`: Map (optional; small numeric/string parameters)

Reserved namespaces in the same top-level map:

- `view`: Map (system-managed view-state)
- `properties`: Map (user-facing gospel assertions; values typed; missing keys remain missing)
- `cached`: Map (cached-derived values; values are Maps containing `value` + `signature`)
- `provenance`: Map (optional; non-semantic provenance)

Readers must ignore unknown top-level keys.

### High-level layout

- Fixed-size **preamble/header** at the front.
- Large **payload region** (matrix/vector binary data) at a stable offset.
- One or more **metadata blocks** appended (append-only updates).

### Header requirements

Header must contain (at minimum):

- magic/version
- endian marker
- payload offset + payload length
- current metadata offset + metadata length
- optional: checksum/CRC for header and metadata blocks (payload checksum optional)

Versioning requirements:

- Header includes a **format version**.
- Metadata blocks include a **metadata encoding version** (may match the header version, but must be explicit).
- Readers must be able to reject unsupported versions deterministically (clear error), without scanning payload.

### Metadata block requirements

Metadata blocks are **self-describing and typed**:

- keys are strings (stable names)
- each value has a type tag (bool/int/float/string/bytes/array/map)
- numeric values include width/sign where relevant

Sparse encoding is mandatory:

- missing key means “unset”, not `False`
- no requirement to materialize defaults in-file

Reserved key namespaces (required):

To keep metadata unambiguous and forward-compatible, the typed metadata map reserves these top-level keys:

- `view`: view-state metadata (system-managed)
- `properties`: user-facing gospel semantic assertions
- `cached` (or `caches`): cached-derived values + validity metadata
- `provenance`: non-semantic provenance (e.g., seed/generation parameters)

Readers must ignore unknown keys.

## Metadata taxonomy (contract; prevents confusion)

R1_STORAGE must support (and clearly separate) **three kinds of metadata**. This is a core clarity requirement: it prevents “random metadata bags” and prevents users/contributors from mixing semantic assertions with system-managed state.

1) **Header / identity metadata** (system-managed)
  - Purpose: define what the object *is* and how to interpret payload bytes.
  - Examples: `rows`, `cols`, `matrix_type`, `data_type`, and any required payload layout descriptor.
  - Notes:
    - These are not “properties” in the user sense.
    - These values are required to correctly load/interpret payload.

Identity/payload layout note:

- `matrix_type` and `data_type` are not always sufficient to describe raw payload layout (e.g., bit-packed layouts, packed triangular storage, row/col-major variants, or future blocked layouts).
- R1_STORAGE must be able to store a minimal **payload layout descriptor** (string/enum + small parameters) so payload interpretation never relies on “magic implied by type names”.

2) **View-state metadata** (system-managed; produced by transforms)
  - Purpose: represent cheap, metadata-only transforms on top of the same payload.
  - Examples: `scalar`, transpose/conjugation/adjoint state.
  - Notes:
    - Users change view-state by applying transforms (e.g., `.T`, conjugation, scaling), not by “asserting” it as a property.
    - View-state participates in cache validity signatures.

3) **User-facing `properties`** (single mapping; two semantic classes)
  - Purpose: a single mapping exposed as `obj.properties`.
  - It contains:
    - **Semantic assertions (gospel):** structure/special-case hints like `is_upper_triangular`, `is_unitary`, `is_identity`. Never truth-validated.
    - **Cached-derived values:** `trace`, `determinant`, `rank`, `norm`, etc. Validity-checked and may be cleared.
  - Critical rule: cached-derived values are user-facing via clean keys (e.g., `trace`) but are persisted explicitly as caches (see below).

This taxonomy is defined semantically by R1_PROPERTIES, but R1_STORAGE is responsible for encoding it unambiguously on disk.

## On-disk encoding conventions (required)

To keep user-facing keys clean while keeping persistence honest, cached-derived values are **not** stored as top-level keys like `cached_trace`.

Instead, metadata uses two top-level sections:

- `properties`: stores gospel semantic assertions (typed; tri-state semantics via key presence).
- `cached` / `caches`: stores cached-derived values (typed) **alongside validity metadata**.

The `view` section is also reserved (system-managed) and is the canonical location for view-state values when they are persisted.

Conceptual shape (illustrative; exact type tags depend on the binary metadata encoding):

```json
{
  "rows": 1000,
  "cols": 1000,
  "matrix_type": "CAUSAL",
  "data_type": "BIT",

  "view": {
    "scalar": 1.0,
    "is_transposed": false,
    "is_conjugated": false
  },

  "properties": {
    "is_unitary": true
  },

  "cached": {
    "trace": {
      "value": 1000.0,
      "signature": {
        "payload_epoch": 17,
        "view_signature": "..."
      }
    }
  }
}
```

Notes:

- The specific serialization of `signature` is an implementation detail, but it must be possible to validate in $O(1)$ during cache lookup.
- The binary typed metadata block may choose not to literally nest `view` as shown above; what matters is that view-state is encoded separately from `properties` and separately from cached-derived values.

R1 decision: `view` is a reserved namespace and is the canonical on-disk location for persisted view-state values. The exact *internal* encoding may vary, but the serialized schema must preserve the separation.

### Update strategy

- Updating metadata must *not* move payload.
- Preferred mechanism: append a new metadata block and atomically update the header pointer.
- A reader uses the header’s “current metadata pointer” to find the authoritative block.

Crash-consistency requirements (must be explicit in implementation):

- Metadata updates must be safe under process crash/power loss.
- The reader must not require scanning the file to recover.

One acceptable approach:

- Maintain two header slots (A/B) with:
  - a monotonically increasing generation counter,
  - the current metadata pointer (offset/length), and
  - a checksum.
- An update writes the new metadata block, then writes the next header slot with a higher generation.
- On load, the reader picks the highest-generation header slot with a valid checksum.

This keeps update/read $O(1)$ and avoids “search backwards for the last valid block”.

Alignment requirements (practical; must be enforced):

- Payload offsets must be aligned to OS mmap granularity (page size).
- Metadata block offsets should also be aligned (at least 8/16 bytes) for simple parsing and predictable IO.

Large-file requirements (must be enforced):

- All offsets/lengths are 64-bit.
- The format must support payloads larger than 4GB on all supported OSes.

Mutable vs append-only metadata (important for practicality):

- Append-only metadata blocks are ideal for occasional updates (save-time metadata, cached-derived values, property edits).
- Some fields change extremely frequently during normal use (e.g., payload content epoch). Persisting those by appending a new metadata block per mutation would bloat files.

R1 note (implemented):

- R1 does **not** persist a per-mutation payload epoch in-file. The header slot fields `hot_offset/hot_length` remain `0`.
- Frequently-changing runtime state (e.g., mutation epochs used for runtime cache invalidation) is maintained **in-memory**.
- Persisted cached-derived validity relies on the persisted snapshot identity (`payload_uuid`) plus a compact view-state signature.

### Snapshot immutability + caches (documented)

This plan does not duplicate snapshot/caching semantics.

Canonical docs:

- [[guides/Storage and Memory]]

## Integration contract with R1_PROPERTIES

- R1_PROPERTIES defines the semantics of `obj.properties` (gospel assertions + cached-derived values).
- Storage must preserve, without scans:
  - key presence vs absence (tri-state semantics via missing keys),
  - typed values,
  - and unknown keys (pass-through / forward compatibility).

Load/save bridging rules (required):

- On load:
  - `properties.*` become entries in `obj.properties`.
  - `cached.*` entries are surfaced as `obj.properties` entries (e.g., `cached.trace` → `obj.properties["trace"]`) only if their dependency signature matches the restored object state; otherwise they are ignored/cleared.
- On save:
  - gospel assertions are written under `properties.*`.
  - cached-derived values are written under `cached.*` with validity metadata.
  - cached-derived values are never written as top-level keys like `cached_trace`.

## Staging / compatibility

- R1 may need a transition period where both formats can be read.
- Writing should be single-file by default once implemented.
- If dual-read exists, it must be explicit and testable (no silent ambiguity).

## Phased execution plan (sizeable; implementation checklist)

This section breaks R1_STORAGE into **large, verifiable phases**. Each phase has:

- **Goal** (what is proven true at the end)
- **Work** (what must be implemented/decided)
- **Deliverables** (artifacts you can point to)
- **Acceptance criteria** (what must pass)

Important: phases are ordered to minimize churn. Do not start a later phase until the earlier phase’s acceptance criteria are met.

### Phase 0 — Contract freeze (format + invariants)

Goal:

- The on-disk contract is frozen enough that implementation can begin without rediscovering format questions mid-flight.

Work (must be decided in writing):

- Exact **binary header layout**:
  - magic bytes, format version, endian marker
  - two header slots (A/B) structure: generation counter, metadata pointer, payload pointer, checksums
  - field widths (must be 64-bit for offsets/lengths) and alignment/padding rules
- Exact **metadata block framing**:
  - metadata block magic/version, length, checksum
  - how unknown keys are skipped without scanning
- Exact **typed metadata encoding v1**:
  - supported types for R1 (bool/int/float/string/bytes/array/map)
  - how numeric widths/sign are represented
  - canonical string encoding (UTF-8)
  - max key length / reasonable limits
- Exact **payload layout descriptor** contract:
  - where it lives (identity/header metadata)
  - what parameters it may contain
  - the rule that payload interpretation never relies on “implied by type name”
- Explicit **read/write policy** for reserved namespaces:
  - `view`, `properties`, `cached`/`caches`, `provenance`
  - what “ignore unknown keys” means for each namespace
- Explicit **crash-consistency rule** (no scan recovery):
  - write ordering (data/metadata/header)
  - what constitutes a valid header slot
- Explicit **large-file + alignment** guarantees:
  - mmap alignment requirements
  - support for >4GB payloads

Deliverables:

- This plan updated with the frozen choices above (no ambiguous “implementation detail” for core layout).
- A short “format summary” section suitable for implementers to copy into code comments.

Acceptance criteria:

- A new contributor can implement a reader/writer without asking format questions.
- The plan’s crash-consistency story is $O(1)$ and does not require “scan backwards for last block”.

### Phase 1 — Minimal container (read/write)

Goal:

- `pycauset.save()` writes the single-file container.
- `pycauset.load()` loads the single-file container.

Work (must be implemented):

- Reject non-container inputs deterministically (fail fast if magic mismatch).
- Implement new writer:
  - write header slot A (or both slots) in an initial “empty metadata” state
  - write payload at an aligned offset (stable)
  - write metadata block (at least identity + view-state) and commit pointer via header slot update
- Implement new reader:
  - choose valid header slot (A/B) by generation + checksum
  - validate referenced metadata block (checksum/length)
  - compute payload offset and pass it to native `_from_storage(...)` exactly as today
- Enforce deterministic failure:
  - invalid header → fail
  - invalid referenced metadata block → fail
  - no “try to find a later block”

Deliverables:

- New container support implemented in the persistence layer (no API changes).
- Only one format is supported.

Acceptance criteria:

- Any file that is not the container format fails deterministically.
- New files are single-file containers and still mmap correctly via stable payload offsets.
- A corrupted header or metadata pointer fails deterministically (no scanning).

### Phase 2 — Typed metadata v1 + taxonomy enforcement

Goal:

- The new container stores and restores the metadata taxonomy unambiguously and sparsely.

Work:

- Encode/decode typed metadata blocks with:
  - reserved namespaces present only when needed
  - missing keys remain missing (never auto-materialize defaults)
  - unknown keys ignored/preserved as appropriate
- Establish the minimal identity/header metadata set that must be persisted for all objects.
- Persist and restore view-state under `view`.

Deliverables:

- Typed metadata block implementation is stable and versioned.
- A documented mapping from in-memory state → on-disk namespaces.

Acceptance criteria:

- Round-trip preserves:
  - key presence vs absence (tri-state semantics via missing keys),
  - typed values,
  - unknown keys (forward compatibility) without breaking load.

### Phase 3 — Cache persistence integration (including inverse)

Goal:

- Cached-derived values are persisted under `cached.*` with validity metadata.
- “Extra blobs” (e.g., an inverse payload) have an R1 home as **independent `.pycauset` objects** referenced from the base snapshot (the sibling object store model), without any archive/member-based packaging.

Work:

- Define how `cached.*` entries are stored (value + signature) in typed metadata.
- Implement load/save bridging:
  - surface valid cached-derived values into `obj.properties` on load
  - write them back under `cached.*` on save
- Replace extra artifacts (e.g., an inverse payload) with a container-native mechanism:
  - either as named typed-metadata bytes entries, or
  - as appended named data blocks referenced by metadata (preferred for large blobs)

#### Definition: “big blob cache” (R1 decision)

A cached-derived value is a **big blob cache** iff persisting it requires referring to the contents of another `PersistentObject` (because the cached value is too large to store directly in typed metadata).

Examples:

- The inverse matrix of a matrix.
- Large factorization artifacts.

Non-examples:

- `trace`, `rank`, `determinant` when represented as small typed values inside `cached.*`.

R1 rule:

- Big blob caches must be persisted as **independent storage objects**.
- The base object stores only a **typed reference** (link) under `cached.*`.

#### Big blob cache protocol (R1 direction; implement safely)

Goal: enable “disk is infinite, compute time is finite” persistence without making the base file fragile.

Storage shape:

- A big-blob cached artifact is stored as its own `.pycauset` container (a normal `PersistentObject`).
- The base object stores a link to it under `cached.<name>` (e.g., `cached.inverse`).

Minimum link fields (typed metadata):

- `ref_kind`: String (`sibling_object_store`)
- `object_id`: String (UUID hex)
- `signature`: Map (validity identity; must be checkable in $O(1)$)

On-disk placement (R1):

- Big-blob objects live next to the base snapshot in `BASE.pycauset.objects/<object_id>.pycauset`.

Signature requirements (no payload scans):

- Must include a persisted snapshot identity for the base payload (e.g., `payload_epoch` or a `payload_uuid`-style identifier that changes when the payload bytes change during persistence).
- Must include view-state identity if view affects the meaning of the cached value (e.g., transpose/scalar).

Crash-consistent write ordering (must not leave dangling half-written references):

1) Write the big-blob object completely (prefer temp name).
2) Make it durable enough for the platform (flush if used).
3) Atomically publish it (rename to final path/id).
4) Append a new metadata block to the base file linking to it.
5) Commit the base metadata pointer via the inactive A/B header slot.

Failure semantics (R1 decision; aligns with Warnings & Exceptions):

- If a big-blob cache link is missing, stale, or points to a corrupt object:
  - treat it as a cache miss (ignore/clear the cached entry),
  - emit a **user-facing warning** (`PyCausetStorageWarning`; no implicit recompute),
  - continue loading the base object.

Deliverables:

- Cached-derived metadata persists in the new format.
- Inverse caching does not depend on any archive/member-based packaging.

Acceptance criteria:

- Cache lookups remain $O(1)$.
- Cached-derived values are never treated as gospel structure.
- If signatures are malformed or stale, cached entries are ignored/cleared.

Additional acceptance criteria (big blob caches):

- The base object never points to a partially-written big-blob object after a crash.
- Missing/corrupt big-blob caches are never implicitly recomputed; regeneration must be explicitly requested by the user.

### Phase 4 — Native/C++ persistence alignment (if applicable in R1)

Goal:

- The native layer can open `.pycauset` files via payload offset/length without any container-implementation assumptions beyond the frozen contract.

Work:

- Identify all places in C++ that assume “raw backing file starts at offset 0” vs “payload has an offset”.
- Ensure that native constructors that accept `(path, offset, ...)` continue to work.
- If native code has its own file writer, either:
  - switch it to write the new format, or
  - explicitly declare Python as the writer for R1 and keep native as read-only for new format.

Deliverables:

- Updated native loader/writer behavior documented in internals.

Acceptance criteria:

- New-format files work for at least the core matrix/vector types on the supported platforms.

### Phase 5 — Hard-break policy (single format)

Goal:

- Pre-alpha policy: when the file format changes, it changes.
- There are no fallback readers, migration paths, or compatibility layers.

Work:

- Ensure error messages clearly distinguish:
  - “magic mismatch / not a `.pycauset` container”
  - “container header invalid”
  - “container metadata invalid”

Deliverables:

- Clear error messages and tests confirming no fallback behavior.

Acceptance criteria:

- No fallback behavior.

### Phase 6 — Testing + debugging (EXTENSIVE; final engineering gate)

Goal:

- Storage is reliable on real machines (Windows included), debuggable under failure, and does not violate the “no scans / stable mmap offsets” constraints.

Work: unit tests (format invariants)

- Header slot selection:
  - valid A/invalid B → choose A
  - invalid A/valid B → choose B
  - both invalid → fail with clear error
- Checksum behavior:
  - corrupt 1 byte in header → reject
  - corrupt 1 byte in metadata block → reject
- Pointer validation:
  - metadata offset points outside file → reject
  - payload offset not aligned → reject (or fail deterministically)
- Sparse semantics:
  - missing key remains missing after round-trip
  - explicit `False` remains explicit

Work: integration tests (real objects)

- Round-trip for representative object types:
  - triangular bit matrix
  - dense bit matrix (rectangular)
  - float matrix
  - integer matrix
  - vectors (int/float/bit) where available
- View-state persistence:
  - transposed + conjugated flags survive save/load
  - scalar survives save/load
- Cache persistence:
  - cached-derived values (trace/determinant/rank/norm) persist and are validated
  - stale signatures are cleared/ignored
- CausalSet persistence:
  - spacetime metadata round-trips
  - underlying matrix remains mmap-backed and correct shape

Work: format mismatch tests

- If magic mismatches, fail fast with a clear error.
- If header is present but invalid (CRC/offsets), fail deterministically.

Work: crash-consistency tests (must not require scanning)

- Simulate an interrupted update sequence:
  - write metadata block but not header pointer → old state loads
  - write header pointer but corrupt new metadata block → load fails deterministically
- Verify that “recovery” is $O(1)$ (choose header slot, validate pointer, stop).

Work: platform/IO tests (Windows pain points)

- Unicode paths (already tested; must continue to pass).
- Nested directories creation.
- Overwrite behavior:
  - saving twice to the same path results in a valid file
- File locking:
  - ensure handles are closed so test cleanup does not fail

Work: performance sanity (non-benchmark gate)

- Confirm new load path does not read payload eagerly.
- Confirm payload offset remains stable and mmap-friendly.

Debugging runbook (must be documented and validated during Phase 6)

- “How to tell what format a file is” (magic bytes; minimal inspection).
- “How to inspect header slot A/B” (fields + checksum).
- “How to inspect metadata block framing” (length/checksum/version).
- “How to debug a failed load” with a step-by-step checklist:
  1) confirm file size
  2) confirm header magic/version
  3) validate chosen header slot checksum
  4) validate metadata pointer range
  5) validate metadata block checksum
  6) confirm payload pointer range/alignment
- Provide at least one developer tool path:
  - either a small debug helper function (Python) or a CLI script that prints header/metadata summary
  - and a test that uses it on a known-good file

Acceptance criteria:

- All storage-related Python tests pass.
- Crash-consistency tests demonstrate $O(1)$ recovery (no scanning).
- Known Windows cleanup/file-lock issues are addressed (tests do not leave files open).

### Phase 7 — Documentation (EXTENSIVE; per Documentation Protocol)

Goal:

- The storage format change has a clear, hard-to-miss doc footprint for users and contributors.

Doc impact assessment (required; classify the change):

- **Internals change:** new persistence container, crash-consistency model, metadata encoding.
- **Behavior:** `.pycauset` is a binary container (not an archive).
- Potential **performance change:** faster/more direct mmap behavior and reduced container overhead.

Work (follow `documentation/project/protocols/Documentation Protocol.md`):

API reference (if public behavior changes):

- Review whether `pycauset.save` / `pycauset.load` need explicit documentation updates (same signature, but different file format).
- If so, update the relevant pages under `documentation/docs/` with:
  - what changed
  - compatibility notes
  - exceptions/failure modes
  - minimal example

Guides (user workflows):

- Update (prefer editing existing) the storage guide(s) to cover:
  - what a `.pycauset` file is now
  - how to move/copy it safely
  - what “mmap-friendly payload” means in practice
  - what users should do if a file is corrupted (and what they cannot do)
- Ensure examples are current and do not reference archive members.

Internals (contributor/maintainer):

- Ensure this plan remains the canonical “how it works” reference and add:
  - a concise “format summary” section
  - explicit invariants and failure modes
  - where the code lives (Python and/or C++)
  - how to extend typed metadata safely (new keys/types)
- Update other internals pages that reference the old archive-style persistence to match reality.

Dev handbook (process changes):

- If build/test workflows change (new scripts/tools), document them under `documentation/dev/`.

Linking + See also (required):

- Add/verify “See also” sections for the key touched docs (3–8 links).
- Use explicit roamlinks paths where possible.

Documentation acceptance criteria (Definition of Done):

- The doc footprint answers:
  - What changed?
  - Who is it for?
  - How do I use it?
  - Constraints and failure modes?
- No stale references to archive-style inspection remain in user-facing docs.
- All updated examples match current APIs.

Corruption and error handling (required):

Corruption and error handling (required):

- If the header checksum fails (or both header slots are invalid), loading must fail with a clear error.
- If a referenced metadata block fails its checksum/length validation, loading must fail deterministically (do not scan for an alternative).
- If metadata is present but semantically incompatible (e.g., a cached-derived signature is malformed), the loader must conservatively ignore/clear the affected cached-derived entries rather than guessing.

Concurrency expectations (explicit policy):

- Multiple readers (read-only mapping) are supported.
- Concurrent mutation of the same file by multiple writers is out of scope unless the implementation introduces explicit file locking.
- If file locking is used, it must be documented and testable (no “sometimes it works”).

## Testing requirements

- Round-trip correctness for payload + metadata (including unset vs explicit False).
- mmap correctness (payload offset correctness) across OSes.
- Append-update correctness (older metadata blocks ignored; pointer respected).
- Large-file performance sanity (no accidental full reads).

Additional minimum tests:

- Crash-consistency simulation for metadata updates (valid header slot selection; no scanning required).
- Corruption handling for invalid header checksum and invalid metadata block checksum.
- Reserved namespace behavior (`view`/`properties`/`cached`/`provenance`) and unknown-key pass-through.

## See also

- `documentation/internals/plans/R1_PROPERTIES_PLAN.md`
- `documentation/internals/plans/completed/R1_PROPERTIES_PLAN.md`
- `documentation/project/protocols/Documentation Protocol.md`
