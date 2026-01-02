# Storage Semantics (Release 1)

This page is the developer-facing hub for how storage works in Release 1. User-facing guidance lives in [[guides/Storage and Memory]], but the mechanics and invariants are captured here.

## Canonical references
- [[guides/Storage and Memory]] (user workflow + container overview)
- [[internals/plans/completed/R1_STORAGE_PLAN.md|R1_STORAGE_PLAN]] (frozen container contract)
- [[internals/plans/completed/R1_PROPERTIES_PLAN.md|R1_PROPERTIES_PLAN]] (metadata semantics and gospel properties)

## Release 1 guarantees (storage)
- Single-file `.pycauset` container, little-endian, with a 4096-byte header holding two slots (A/B). The valid slot with the highest generation is active; CRC mismatch or out-of-range pointers fail deterministically.
- Payload offset is aligned (≥4096) and never moves after creation. Metadata offset is aligned (≥16). Updates append metadata and flip the inactive header slot; no scanning is required.
- Typed metadata is a single sparse map with reserved namespaces: `view`, `properties`, `cached`, `provenance`, plus identity/header keys (`rows`, `cols`, `matrix_type`, `data_type`, `payload_layout`). Missing keys stay missing (tri-state semantics).
- Block matrices persist as a base container plus a sidecar `<name>.pycauset.blocks/`; the manifest pins child `payload_uuid` values to avoid mixed snapshots.

## Snapshot + mutation semantics
- `.pycauset` files are immutable snapshots; `load()` returns a snapshot-backed object.
- Mutations use copy-on-write working copies; payload bytes in the snapshot are not overwritten implicitly.
- Frequently changing runtime epochs stay in memory; header `hot_offset`/`hot_length` remain 0 in R1.
- Saving writes a new snapshot (base file, plus sidecar for block matrices). Overwrites replace the header slot generation and metadata pointer but keep the payload offset stable.

## Cache semantics (small + big blobs)
- `payload_uuid` identifies the persisted payload snapshot; `view_signature` captures view-state (`scalar`, `is_transposed`, `is_conjugated`). Both are used to validate cached-derived values without scanning.
- Small cached-derived values live under `cached.*` with a validity signature and surface into `obj.properties` on load when valid.
- Big-blob caches are persisted as independent `.pycauset` objects in `BASE.pycauset.objects/<object_id>.pycauset`; the base metadata stores a typed reference (`ref_kind = sibling_object_store`, `object_id`, `signature`). Missing/unreadable/stale references raise `PyCausetStorageWarning` and are treated as cache misses (ignored), with **no implicit recomputation**.

## Crash-consistent update path
1) Append the new typed metadata block (framing: `PCMB`, block_version=1, encoding_version=1, payload_crc32 validated).
2) Flush if the implementation uses explicit flush.
3) Write the inactive header slot with `generation = active + 1` and the new metadata pointer; include slot CRC32.
4) Optionally flush the header region.

Load is $O(1)$: pick the highest-generation valid slot, validate pointers/CRCs, read metadata, mmap payload at `payload_offset`.

## Debugging checklist
- Confirm magic/version/endian in the preamble (`PYCAUSET`, version 1, little-endian).
- Validate header slots: CRC, pointer ranges, alignment (payload 4096-aligned, metadata 16-aligned), choose highest valid generation.
- Validate metadata framing: `PCMB`, versions supported, payload length within file, CRC32 matches.
- Validate payload pointer: `payload_offset + payload_length` within file; payload is mmap-friendly at that offset.
- Developer helper: `python/pycauset/_internal/storage_debug.py::summarize_container(path)` (used by `tests/python/test_storage_debug_tool.py`).

## See also
- [[guides/Storage and Memory]]
- [[internals/MemoryArchitecture.md|MemoryArchitecture]]
- [[internals/Memory and Data.md|Memory and Data]]
- [[project/protocols/Documentation Protocol.md|Documentation Protocol]]
