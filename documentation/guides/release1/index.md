# Release 1 (R1): What Shipped

Release 1 is the first “stable foundation” slice of PyCauset: rectangular matrices, a real dtype system (including complex floats), a persistent on-disk container, and a semantic `properties` mechanism that can change algorithm choices.

This section summarizes the *implemented user- and contributor-facing behavior*, not the planning documents.

## In this section

- **Shapes & NxM matrices**: [[guides/release1/shapes.md|R1 Shapes (NxM Matrices)]]
- **Persistence & snapshots**: [[guides/release1/storage.md|R1 Storage (Persistence Container)]]
- **Semantic properties**: [[guides/release1/properties.md|R1 Properties (Semantic Metadata)]]
- **DTypes, promotion, overflow**: [[guides/release1/dtypes.md|R1 DTypes (Integers, Float16, Complex)]]
- **Linear algebra endpoints**: [[guides/release1/linalg.md|R1 Linear Algebra (Core Ops)]]

## See also

- [[guides/NxM Support.md|NxM Support Status]]
- [[guides/Storage and Memory.md|Storage and Memory]]
- [[docs/index.md|API Reference]]
- [[internals/index.md|Internals]]
- [[dev/index.md|Dev Handbook]]

## Completed plan coverage (traceability)

This section is the “are we missing anything?” crosswalk from the **completed R1 plans** to the **front-end documentation**.

- **R1_SHAPES** → [[guides/release1/shapes.md|R1 Shapes]] (and [[guides/NxM Support.md|NxM Support Status]])
	- Plan artifact: [[internals/plans/completed/R1_SHAPES_PLAN.md|R1_SHAPES_PLAN]]
- **R1_STORAGE** → [[guides/release1/storage.md|R1 Storage]] (canonical: [[guides/Storage and Memory.md|Storage and Memory]])
	- Plan artifact: [[internals/plans/completed/R1_STORAGE_PLAN.md|R1_STORAGE_PLAN]]
- **R1_PROPERTIES** → [[guides/release1/properties.md|R1 Properties]] (API footprint in [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]] / [[docs/classes/vector/pycauset.VectorBase.md|VectorBase]])
	- Plan artifact: [[internals/plans/completed/R1_PROPERTIES_PLAN.md|R1_PROPERTIES_PLAN]]
- **DTYPE_COMPLEX_OVERFLOW** → [[guides/release1/dtypes.md|R1 DTypes]] (canonical: [[internals/DType System.md|DType System]])
	- Plan artifact: [[internals/plans/completed/DTYPE_COMPLEX_OVERFLOW_PLAN.md|DTYPE_COMPLEX_OVERFLOW_PLAN]]
- **R1_LINALG** → [[guides/release1/linalg.md|R1 Linear Algebra]]
	- Plan artifact: [[internals/plans/completed/R1_LINALG_PLAN.md|R1_LINALG_PLAN]]
- **Restructure** → [[dev/index.md|Dev Handbook]] (starting at [[dev/Restructure Plan.md|Restructure Plan]])
	- Plan artifact: [[internals/plans/completed/Restructure Plan.md|Restructure Plan (execution record)]]
