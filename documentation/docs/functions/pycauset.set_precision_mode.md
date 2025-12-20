# pycauset.set_precision_mode

```python
pycauset.set_precision_mode(mode: str) -> None
```

Set the current thread-local promotion precision mode.

## Parameters

- `mode`: Either `"lowest"` or `"highest"`.

## Description

PyCauset has an explicit promotion policy that controls how result dtypes are chosen when mixing dtypes.
This function selects the policy used by the **current thread**.

- `"lowest"`: choose the smallest reasonable storage dtype (scale/storage-first default).
- `"highest"`: choose the highest-precision storage dtype available within the participating operand ranks.

Notes:

- This controls **storage dtype selection** (the dtype of the resulting matrix/vector).
- It does not directly control accelerator internal compute dtype.

## Example

```python
import pycauset as pc

pc.set_precision_mode("highest")
```

## See also

- [[docs/functions/pycauset.get_precision_mode.md|pycauset.get_precision_mode]]
- [[docs/functions/pycauset.precision_mode.md|pycauset.precision_mode]]
- [[docs/functions/pycauset.divide.md|pycauset.divide]]
