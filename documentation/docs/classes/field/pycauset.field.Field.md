# pycauset.field.Field

```python
class Field(kind="scalar", *, mass=0.0, spin=0, scheme=None)
```

Set-independent field content (R2): the field's species — `kind`, `mass`, `spin`, and
discretization `scheme` — independent of any background. You apply it to a background
with `.on()`.

## Parameters

*   **kind** (*str*): The species. Only `"scalar"` is implemented; unknown kinds raise `NotImplementedError`.
*   **mass** (*float*): The field mass. Defaults to 0.0 (massless).
*   **spin** (*int*): Reserved for future species. Defaults to 0.
*   **scheme**: Reserved for future discretization schemes.

## Methods

### on

```python
def on(self, background) -> CorrelatedField | ContinuumCorrelatedField
```

Applies the field to a background:

*   `phi.on(causet)` → a [[docs/classes/field/pycauset.field.CorrelatedField.md|CorrelatedField]] (the field + its Green's functions and vacuum two-point on that causet).
*   `phi.on(spacetime)` → a [[docs/classes/field/pycauset.field.ContinuumCorrelatedField.md|ContinuumCorrelatedField]] (closed-form Green's functions on a continuum Minkowski spacetime).

## Example

```python
import pycauset as pc

phi = pc.field("scalar", mass=1.5)   # the Field, background-independent
Q   = phi.on(causet)                 # a CorrelatedField on the causet
```

## See also

- [[docs/classes/field/pycauset.field.CorrelatedField.md|CorrelatedField]]
- [[docs/classes/field/pycauset.field.State.md|State]]
- [[docs/functions/pycauset.field.md|pc.field]]
- [[guides/Field Theory.md|Field Theory guide]]
