# Restructure (what changed)

This page summarizes the executed codebase restructure work that landed alongside the Release 1 foundations.

The goal is contributor clarity: where things live, what the canonical build/test workflows are, and which tooling prevents drift.

## What changed

- **Developer handbook exists and is canonical:** the repo now has a dedicated `documentation/dev/` section covering build, bindings, testing, and hygiene.
- **Python internals are modularized:** implementation code lives under `python/pycauset/_internal/`, while the public surface remains `pycauset.*`.
- **Bindings are modularized:** `src/bindings.cpp` is a thin entrypoint and binding logic is split across `src/bindings/*` by subsystem.
- **Drift prevention tooling exists:** `tools/check_native_exports.py` helps catch mismatches between Python expectations and native exports.

## How to navigate the codebase now

- Python public facade: `python/pycauset/__init__.py`
- Python implementation modules: `python/pycauset/_internal/`
- Native bindings: `src/bindings/`
- Native core and compute: `src/` and `include/pycauset/`

See [[dev/Codebase Structure.md|Codebase Structure]] for the canonical map.

## Build and test are documented workflows

The documented source-of-truth build workflow is pip/scikit-build-core.

See:

- [[dev/Build System.md|Build System]]
- [[dev/Testing & Benchmarks.md|Testing & Benchmarks]]

## Approval gate note

Some additional restructure work was intentionally deferred behind an explicit approval gate. The execution record and remaining proposals live in:

- [[internals/plans/completed/Restructure Plan.md|Internals: Restructure execution record]]

## See also

- [[dev/index.md|Dev Handbook]]
- [[dev/Python Internals.md|Python Internals]]
- [[dev/Bindings & Dispatch.md|Bindings & Dispatch]]
- [[dev/Repository Hygiene.md|Repository Hygiene]]
- [[project/protocols/Documentation Protocol.md|Documentation Protocol]]
