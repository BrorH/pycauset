# pycauset.CausalSpacetime

```python
class pycauset.CausalSpacetime
```

The native abstract base type for spacetime manifolds in PyCauset.

## Description

In Release 1 this class is exposed as a native type marker only: it has no public
Python constructor and no bound methods. The concrete spacetime regions are the
Minkowski classes (see below), which are the objects you actually construct and pass
to `pycauset.causet(...)`.

## Concrete spacetimes

* [[docs/classes/spacetime/pycauset.spacetime.MinkowskiBox.md|pycauset.spacetime.MinkowskiBox]]
* [[docs/classes/spacetime/pycauset.spacetime.MinkowskiCylinder.md|pycauset.spacetime.MinkowskiCylinder]]
* [[docs/classes/spacetime/pycauset.spacetime.MinkowskiDiamond.md|pycauset.spacetime.MinkowskiDiamond]]

## See also

* [[docs/functions/pycauset.causet.md|pycauset.causet]]
* [[docs/classes/spacetime/pycauset.CausalSet.md|pycauset.CausalSet]]
