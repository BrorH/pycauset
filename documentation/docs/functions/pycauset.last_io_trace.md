# pycauset.last_io_trace

```python
pycauset.last_io_trace(op: str | None = None) -> dict | None
```

Return the trace of the most recent traced call, optionally the most recent call of one
operation. Traced calls are the `pycauset.<op>(...)` functions (`matmul`, `invert`, `solve`,
the decompositions and norms, `trace`, `determinant`, the bitwise functions, ...); operators
such as `a @ b` are not traced.

A trace holds facts only:

| Key | Meaning |
|---|---|
| `op` | the operation name |
| `operands` | one dict per input: `shape`, `dtype`, `storage` (`"memory"`, `"file"` or `"temp"`), `backing_file` |
| `kernels` | every kernel the engine ran for the call, in order (engine labels such as `"cpu.matmul.f64"`) |
| `timestamp` | `time.time()` at the start of the call |

Whether an operation streams is the engine's decision: it streams when the operation does not
fit the memory governor's RAM budget (see [[guides/Storage and Memory|Storage and Memory]]).
A traced call that calls another traced function records only the outer call.

## Parameters

*   **op** (*str, optional*): the most recent trace of this operation (for example
    `"matmul"`). `None`: the most recent trace of any operation.

## Returns

*   **dict | None**: a copy of the trace record, or `None` when there is none.

## See also

*   [[pycauset.clear_io_traces|pycauset.clear_io_traces]]
