# PyCauset Container Format

This page specifies the on-disk container format for `.pycauset` files.

If you want a conceptual and user-oriented explanation (how to save/load safely, copying rules, mental model), start here:

- [[guides/Storage and Memory]]

## Goals

The container format is designed to satisfy three constraints at once:

1) **mmap-friendly payload**: large payload bytes live at a stable, aligned offset.
2) **deterministic load**: loading selects an active header slot in $O(1)$.
3) **crash-consistent metadata updates**: metadata updates are append-only and committed by flipping a header slot.

## High-level layout

1) Fixed **4096-byte header** with a preamble + two header slots (A/B).
2) **Payload region** at `payload_offset` (aligned to 4096 bytes).
3) **Append-only metadata block** at `metadata_offset` (aligned to 16 bytes).

## Endianness

- Containers are **little-endian only**.
- A header endian marker allows fast failure on unsupported endianness.

## Alignment

- `payload_offset` is aligned to **4096 bytes**.
- `metadata_offset` is aligned to **16 bytes**.

## Fixed header (4096 bytes)

The file begins with a fixed header region:

- A 16-byte preamble.
- Two 128-byte header slots (A and B).
- The remainder reserved (zero in the current format).

### Preamble (offset 0)

| Field | Type | Notes |
|---|---:|---|
| `magic` | 8 bytes | ASCII `PYCAUSET` |
| `format_version` | u32 | current = 1 |
| `endian` | u8 | 1 = little-endian |
| `header_bytes` | u16 | current = 4096 |
| `reserved0` | u8[1] | must be 0 |

### Header slots (A and B)

Each slot is 128 bytes and stores the authoritative pointers.

| Field | Type | Notes |
|---|---:|---|
| `generation` | u64 | monotonic; higher wins |
| `payload_offset` | u64 | aligned to 4096 |
| `payload_length` | u64 | bytes |
| `metadata_offset` | u64 | aligned to 16 |
| `metadata_length` | u64 | bytes |
| `hot_offset` | u64 | 0 in v1 |
| `hot_length` | u64 | 0 in v1 |
| `slot_crc32` | u32 | CRC32 of the first 7 fields (56 bytes) |
| `slot_reserved` | u8[68] | must be 0 |

Slot validity (v1):

- `slot_crc32` matches
- offsets/lengths are in-range for file size
- alignment constraints satisfied

Active slot selection:

- Choose the valid slot with the highest `generation`.
- If neither slot is valid, loading fails.

## Payload region

The payload is raw bytes suitable for memory mapping:

- Starts at `payload_offset`.
- Spans `payload_length` bytes.
- Interpretation is defined by identity metadata (shape, dtype, matrix type, `payload_layout`).

## Metadata blocks (append-only)

Metadata is stored as blocks after the payload. The active header slot points to the authoritative block.

### Metadata framing (at `metadata_offset`)

| Field | Type | Notes |
|---|---:|---|
| `block_magic` | 4 bytes | ASCII `PCMB` |
| `block_version` | u32 | v1 = 1 |
| `encoding_version` | u32 | typed-metadata encoding version; v1 = 1 |
| `reserved0` | u32 | must be 0 |
| `payload_length` | u64 | bytes of encoded metadata payload |
| `payload_crc32` | u32 | CRC32 of encoded metadata payload |
| `reserved1` | u32 | must be 0 |
| `payload` | bytes | length = `payload_length` |

If framing or CRC fails, loading fails deterministically.

### Typed metadata map (v1)

The encoded metadata payload is a single top-level map with string keys.

Reserved namespaces:

- identity/header keys: `rows`, `cols`, `matrix_type`, `data_type`, `payload_layout`, `payload_uuid`, ...
- `view`: system-managed view-state
- `properties`: user-facing gospel assertions
- `cached`: cached-derived values plus validity metadata
- `provenance`: optional non-semantic provenance

Readers ignore unknown keys.

## Crash-consistent metadata update rule

To update metadata without scanning:

1) Append the new metadata block to the end of the file.
2) Ensure it is fully written (and flushed if applicable).
3) Write the inactive header slot with `generation = active.generation + 1` and the new metadata pointer.
4) Optionally flush the header region.

This guarantees deterministic $O(1)$ load and never moves the payload region.

## Debugging notes

### When a `.pycauset` file fails to load

1) Confirm magic `PYCAUSET` and version.
2) Inspect header slot A/B:
	- CRC valid?
	- offsets/lengths in-range?
	- alignments satisfied?
	- which slot is active?
3) Validate the metadata block framing and CRC.

Developer tooling:

- `python/pycauset/_internal/storage_debug.py` exposes `summarize_container(path)`.
