"""User notes on a matrix or vector: ``obj.properties``.

A free-form mapping of string keys to values the user attaches to an object (where the data
came from, a seed, a label). It is saved with the object (``pc.save``) and restored on load.
PyCauset never reads it: structure lives in the class (TriangularMatrix, SymmetricMatrix,
HermitianMatrix, DiagonalMatrix, ...), and a result never depends on a note. Notes are not
carried to views or results (``A.T`` is another object; a note about A need not hold for it).

The boolean structure flags this module used to hold (``is_upper_triangular``,
``is_hermitian``, ``is_zero``, ...; validation, propagation through views, invalidation on
writes, a C++ bitmask) are removed (review F-092): the structured classes guarantee what the
flags only asserted.
"""
from __future__ import annotations

from typing import Any

_ATTR = "_pycauset_properties"


def get_properties(obj: Any) -> dict[str, Any]:
    """The object's notes (a live dict: changes are kept)."""
    store = getattr(obj, _ATTR, None)
    if not isinstance(store, dict):
        store = {}
        setattr(obj, _ATTR, store)
    return store


def set_properties(obj: Any, mapping: Any) -> None:
    """Replace the object's notes with ``mapping`` (string keys)."""
    new = dict(mapping or {})
    for key in new:
        if not isinstance(key, str):
            raise TypeError("properties keys must be strings")
    setattr(obj, _ATTR, new)


def install(classes: list[Any]) -> None:
    """Give each class an ``obj.properties`` attribute (get: the dict; set: replace it)."""
    prop = property(get_properties, set_properties, doc=__doc__.split("\n\n")[0])
    for cls in classes:
        if cls is not None:
            cls.properties = prop
