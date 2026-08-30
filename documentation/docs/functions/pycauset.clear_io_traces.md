# pycauset.clear_io_traces

```python
pycauset.clear_io_traces()
```

Clear all recorded IO traces.

IO traces are lightweight observability records produced during operations (see
[[pycauset.last_io_trace]]). Clearing them frees the accumulated records; this is a
debug/observability helper, not a runtime requirement.
