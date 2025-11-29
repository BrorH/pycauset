```
pycauset.save
```
Global boolean flag controlling whether temporary matrices are persisted to disk after the program exits.

- Type: `bool`
- Default: `False`

If set to `True`, matrices created without an explicit `saveas` path (which are normally treated as temporary) will be preserved in the `.pycauset/` directory. If `False`, they are deleted upon program termination.
