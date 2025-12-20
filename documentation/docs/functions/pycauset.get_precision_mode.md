# pycauset.get_precision_mode

```python
pycauset.get_precision_mode() -> str
```

Return the current thread-local promotion precision mode.

## Returns

- `str`: Either `"lowest"` or `"highest"`.

## Example

```python
import pycauset as pc

mode = pc.get_precision_mode()
print(mode)
```

## See also

- [[docs/functions/pycauset.set_precision_mode.md|pycauset.set_precision_mode]]
- [[docs/functions/pycauset.precision_mode.md|pycauset.precision_mode]]
