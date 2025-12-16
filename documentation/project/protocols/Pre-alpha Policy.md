# Pre-alpha Policy (Scope + Approval Gates)

PyCauset is currently **pre-alpha** and effectively single-maintainer. There is no external userbase to preserve yet.

- Backward compatibility is **not** a hard constraint right now.
- Breaking changes to the Python surface and/or architecture are allowed **when they improve the overall approach**.
- Approval gate: before changing the public Python surface (names/semantics) or making a large architectural shift, propose the change + tradeoffs and wait for explicit approval.
- If a breaking change is approved, update tests and documentation in the same change set (don’t leave the repo in a “half-migrated” state).
