# Storage Semantics (Release 1)

This page is the developer-facing hub for how storage works in Release 1. User-facing guidance lives in [[guides/Storage and Memory]], but the mechanics and invariants are captured here.

## Canonical references
- [[guides/Storage and Memory]] (user workflow + container overview)
- [[internals/PyCauset Container Format|PyCauset Container Format]] (the container specification)

## Release 1 guarantees (storage)
- Single-file `.pycauset` container, little-endian, with a 4096-byte header holding two slots (A/B). The valid slot with the highest generation is active; CRC mismatch or out-of-range pointers fail deterministically.
- Payload offset is 4096 and metadata offset 16-aligned. Each save writes a whole new file and
  renames it over the target; nothing is updated in place.
- Typed metadata is a single map (keys in [[internals/PyCauset Container Format|Container Format]]).
- Block matrices persist as a base container plus a sidecar `<name>.pycauset.blocks/`; the manifest pins child `payload_uuid` values to avoid mixed snapshots.

## Snapshot + mutation semantics
- `.pycauset` files are immutable snapshots; `load()` returns a snapshot-backed object: the
  payload is mapped copy-on-write (MAP_PRIVATE / FILE_MAP_COPY), so a write makes that page
  private and never reaches the file (F-093).
- Saving writes a uniquely named hidden sibling (`.<name>.XXXX.part`), fsyncs it, renames it
  over the target (atomic) and fsyncs the directory: an object loaded from the target keeps
  reading its mapping of the old file, and a crash leaves either the old or the new file.
  On Windows a loaded file cannot be replaced while it is mapped (review 3.9 C-10).
- Header `hot_offset` / `hot_length` are 0.
- Block matrices: a base container plus the sidecar directory, committed by directory renames.

Load is O(1): pick the highest-generation valid slot, validate pointers and CRCs, read
metadata, map the payload at `payload_offset`.

## Debugging checklist
- Confirm magic/version/endian in the preamble (`PYCAUSET`, version 1, little-endian).
- Validate header slots: CRC, pointer ranges, alignment (payload 4096-aligned, metadata 16-aligned), choose highest valid generation.
- Validate metadata framing: `PCMB`, versions supported, payload length within file, CRC32 matches.
- Validate payload pointer: `payload_offset + payload_length` within file; payload is mmap-friendly at that offset.

## See also
- [[guides/Storage and Memory]]
- [[internals/Memory and Data.md|Memory and Data]]
- [[project/protocols/Documentation Protocol.md|Documentation Protocol]]
