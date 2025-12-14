# Agent Bootstrap (Safe Onboarding)

This page is designed to get a fresh AI agent (or a new human contributor) up to speed **safely**.

## The safest way to onboard an AI agent

### Recommended approach

1. **Point the agent to this folder**: `documentation/dev/`.
2. **Paste the “Bootstrap Prompt”** below as the first message.
3. Require the agent to:
   - summarize the mantras in its own words,
   - propose a plan,
   - ask clarifying questions,
   - and **wait for approval** before making code changes.

This reduces the risk of the agent “helpfully” making changes that violate core design principles.

### Bootstrap Prompt (copy/paste)

You can paste this as the first message to an AI agent:

```
You are working on the PyCauset repo.

Goal: Make progress while preserving the project’s core philosophy.

Hard constraints:
- PyCauset is “NumPy for causal sets”: users interact with top-level Python APIs; performance/dispatch/storage optimizations are automatic and behind the scenes.
- Do not move user-facing APIs behind submodules like pycauset.physics.*. Internal code can be reorganized, but the public surface stays pycauset.*.
- Be hesitant: ask questions when uncertain.
- Do not write code or change files until I explicitly approve.

Onboarding tasks (read-only):
1) Read:
   - documentation/dev/index.md
   - documentation/dev/Restructure Plan.md
   - documentation/project/Philosophy.md
   - documentation/project/Protocols.md
   - documentation/internals/index.md
2) Summarize in <10 bullets:
   - the mantras/invariants,
   - the current architecture,
   - the highest-risk areas (where changes can break things).
3) Propose a step-by-step plan for the next task and ask 1–3 clarifying questions.
4) Wait.
```

## What an agent should learn first (project invariants)

- **Public API invariant:** users should call `pycauset.Matrix`, `pycauset.CausalMatrix`, `pycauset.matmul`, etc. Internal folders/modules may change; the top-level user API should remain stable.
- **Tiered storage:** small objects behave like NumPy; large objects spill to disk automatically.
- **Anti-promotion:** avoid silently inflating dtypes.
- **Direct vs streaming path:** centralized decision logic (avoid per-call ad-hoc heuristics).
- **Cooperative Compute Architecture:** solvers can emit I/O hints; I/O layer prefetches/optimizes.

## What to do if an agent is unsure

- Stop and ask a precise question.
- Offer 2–3 options with tradeoffs.
- Prefer the simplest change that preserves invariants.
