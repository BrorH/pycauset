# Documentation Protocol

This protocol exists so documentation stays **hard to miss**, **hard to rot**, and **easy to expand**.

It is written for humans first. Avoid "writer meta" (how the doc was produced, constraints that only mattered during drafting, etc.). Put the reader’s needs first.

---

## 1) The rule: every change must have a doc footprint

If you change behavior, add a feature, or add a new type/op, you must leave a doc footprint that answers:

- What changed?
- Who is it for (user vs contributor)?
- How do I use it?
- What are the constraints and failure modes?

### Doc impact assessment (required)

Before you touch docs, classify the change:

- **API change** (new/changed public function/class/parameter)
- **Behavior change** (same API but different semantics)
- **Performance change** (new fast-path, new routing, new thresholds)
- **Internals change** (new storage format, new kernel path, new invariants)

Then apply the mapping below.

---

## 2) Where to document (mapping)

### A) API reference (`documentation/docs/`) — mandatory for user-facing surfaces

Add/update:

- `docs/classes/` if you add/modify a class.
- `docs/functions/` if you add/modify a public function.
- `docs/parameters/` if you add a new config parameter or global knob.

Minimum expectation for API reference pages:

- Signature + parameters
- Return type(s)
- Exceptions / warnings
- At least one example that actually runs

### B) Guides (`documentation/guides/`) — mandatory for “how do I use this?”

Guides are where we teach workflows.

Rules:

- Prefer **modifying an existing guide** over adding a new one.
- Add a new guide only if it represents a new domain.

Minimum expectation for guides:

- A problem statement (“what you’re trying to do”)
- A minimal example
- One realistic example (with caveats)
- Links to the relevant API reference pages

### C) Internals (`documentation/internals/`) — mandatory for architecture/backend changes

Internals are for contributors and future AI agents.

Minimum expectation for internals:

- Data model / invariants
- Where in the codebase the behavior lives
- How to extend it safely
- Common failure modes and debugging tips

### D) Dev handbook (`documentation/dev/`) — mandatory for process / contributor UX

Use dev handbook pages when:

- onboarding needs to change,
- build/test workflows change,
- “how to do work safely” changes.

---

## 3) Linking protocol (MkDocs roamlinks)

MkDocs supports wiki links. We use them aggressively, but in a **controlled** way.

### Use explicit links (preferred)

Prefer explicit, stable links that include the path, so the target is unambiguous:

- `[[docs/functions/pycauset.matmul.md|pycauset.matmul]]`
- `[[docs/classes/matrix/pycauset.Matrix.md|pycauset.Matrix]]`
- `[[internals/DType System|internals/DType System]]`

Note: `mkdocs-roamlinks-plugin` treats any `.` in the filename as “has an extension”, so for targets like `pycauset.matmul` / `pycauset.Matrix` you should include the `.md` suffix.

### Avoid ambiguous links

Avoid short wiki-links like `Matrix`, `Installation`, etc. unless you are sure there is only one plausible target.

### Add “See also” sections

Most pages should end with a short “See also” list of 3–8 links. This is what makes the docs scale.

---

## 4) Quality bar (what “good docs” looks like)

### Reader-first openings

Start with what the reader wants to do, not properties of the document.

Good:

- “This guide shows how to store disk-backed matrices and control the storage directory.”

Bad:

- “This document is the canonical source of truth and is not time-based.”

### Staleness checks (required)

When editing a page, quickly verify:

- Does it mention files that no longer exist?
- Does it mention APIs that no longer exist?
- Does it promise performance behavior that isn’t enforced?

If you can’t verify something easily, rewrite it to be:

- specific but testable (“runs through AutoSolver routing”), or
- scoped (“GPU support is available for selected operations; see …”), or
- explicitly marked as a plan (link to a plan doc).

### Avoid hyper-local details

Avoid including details that were only true during a single development session (temporary constraints, incidental path names, one-off benchmark numbers) unless they are truly stable.

---

## 5) Definition-of-done checklist (docs)

Before marking a task complete:

1. [ ] API reference updated (if public)
2. [ ] Guide updated (if user-facing)
3. [ ] Internals updated (if backend/architecture)
4. [ ] Cross-links added (minimum: 3 relevant links)
5. [ ] Examples are correct and match the current API
6. [ ] No “writer meta” at the top of the page

---

## 6) Templates (copy/paste)

### Template: new API reference page

- What it is
- Signature
- Parameters
- Returns
- Exceptions/warnings
- Examples
- See also

### Template: new guide section

- Goal
- Minimal example
- Practical example
- Pitfalls
- See also
