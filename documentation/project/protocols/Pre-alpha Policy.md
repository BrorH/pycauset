# Pre-alpha Policy (Scope + Approval Gates)

PyCauset is currently **pre-alpha** and effectively single-maintainer. There is no external userbase to preserve yet.

- Backward compatibility is **not** a hard constraint right now.
- Breaking changes to the Python surface and/or architecture are allowed **when they improve the overall approach**.
- Approval gate: before changing the public Python surface (names/semantics) or making a large architectural shift, propose the change + tradeoffs and wait for explicit approval.
- If a breaking change is approved, update tests and documentation in the same change set (don’t leave the repo in a “half-migrated” state).

## Removals policy ("deprecation" = purge)

PyCauset uses a purge policy:

- If we decide something should be removed, we remove it fully.
- We do not keep deprecated aliases or “deprecated but still present” codepaths.
- Documentation should never say “deprecated”: it should reflect the current reality.

If a removal breaks internal callers, update them in the same change.
