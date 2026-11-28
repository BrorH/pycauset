# src/: the C++ engine (implementation)

Headers are in `include/pycauset/` with the same layout. Map and call flow:
`documentation/internals/Architecture.md`; procedures:
`documentation/project/protocols/Adding Operations.md` and `Adding Types.md`.

| Folder | What lives here |
|---|---|
| `core/` | storage (PersistentObject, MemoryMapper, MemoryGovernor), ObjectFactory, the structure and dtype resolvers, the kernel log |
| `math/` | the entry function of every operation (structure rules, normaliser, allocation) |
| `compute/` | `ComputeContext` (holds the device); `cpu/` the kernels (out of core: on mapped storage) |
| `causet/` | the sprinkler for the native Minkowski diamond |
| `bindings.cpp`, `bindings/` | the pybind11 module `pycauset._pycauset` |

Not here: Python code (`python/pycauset/`), parked code (`experimental/`). The engine never
prints and never falls back to NumPy.
