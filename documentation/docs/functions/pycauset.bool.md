# pycauset.bool

```python
pycauset.bool
```

The `"bool"` dtype token, usable anywhere a dtype string is accepted.

```python
import pycauset

m = pycauset.matrix(((True, False), (False, True)), dtype=pycauset.bool)
```

Equivalent to passing `dtype="bool"`. Boolean matrices use bit-packed storage.

## See also

*   [[pycauset.matrix|pycauset.matrix]]
*   [[internals/DType System.md|DType System]]
