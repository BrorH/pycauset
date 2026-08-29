# pycauset.field

```python
pycauset.field(kind="scalar", *, mass=0.0, spin=0, scheme=None) -> Field
```

String-factory sugar for the R2 field model: `pc.field("scalar", mass=…)` returns a set-independent [[docs/classes/field/pycauset.field.Field.md|Field]]. Unknown `kind` strings raise `NotImplementedError` (never guessed).

`pycauset.field` is a callable module, so both `pc.field("scalar", …)` and `from pycauset.field import ScalarField` work.

## Parameters

*   **kind** (*str*): The species. Only `"scalar"` is implemented.
*   **mass** (*float*): The field mass.
*   **spin** (*int*): Reserved for future species.
*   **scheme**: Reserved for future discretization schemes.

## Returns

*   **Field**: The set-independent field. Apply it with `phi.on(causet)` / `phi.on(spacetime)`.

## Example

```python
import pycauset as pc

phi = pc.field("scalar", mass=1.5)
Q = phi.on(pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=42))
W = Q.wightman()
```

## See also

- [[docs/classes/field/pycauset.field.Field.md|Field]]
- [[docs/classes/field/pycauset.field.CorrelatedField.md|CorrelatedField]]
- [[guides/Field Theory.md|Field Theory guide]]
