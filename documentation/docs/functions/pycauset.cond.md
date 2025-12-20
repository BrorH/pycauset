# pycauset.cond

```python
pycauset.cond(a, p=None)
```

Compute a condition number estimate.

## Parameters

* **a** (*MatrixBase*): Input matrix.
* **p**: Currently not supported (must be `None`).

## Returns

* **float**: A condition number estimate computed as `norm(a) * norm(invert(a))`.
