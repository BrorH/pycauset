from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any

_PROPERTIES_ATTR = "_pycauset_properties"

# Cached-derived keys (never semantic structure) that must be cleared on any
# payload mutation.
_CACHED_DERIVED_KEYS: tuple[str, ...] = (
    "trace",
    "determinant",
    "rank",
    "norm",
    "sum",
    "eigenvalues",
    "eigenvectors",
)


def _ensure_store(obj: Any) -> dict[str, Any]:
    store = getattr(obj, _PROPERTIES_ATTR, None)
    if store is None or not isinstance(store, dict):
        store = {}
        try:
            setattr(obj, _PROPERTIES_ATTR, store)
        except Exception:
            # If the native object cannot accept attributes, fall back to a transient dict.
            # (Persistence will simply not round-trip these.)
            return {}
    return store


def _get_store(obj: Any) -> dict[str, Any]:
    # Internal helper: returns the underlying dict store (no proxy).
    return _ensure_store(obj)


def _get_shape(obj: Any) -> tuple[int, int] | None:
    try:
        return int(obj.rows()), int(obj.cols())
    except Exception:
        try:
            shape = getattr(obj, "shape", None)
            if isinstance(shape, tuple) and len(shape) == 2:
                return int(shape[0]), int(shape[1])
        except Exception:
            pass
    return None


_BOOL_LIKE_KEYS: frozenset[str] = frozenset(
    {
        "is_zero",
        "is_identity",
        "is_permutation",
        "is_diagonal",
        "is_upper_triangular",
        "is_lower_triangular",
        "has_unit_diagonal",
        "has_zero_diagonal",
        "is_symmetric",
        "is_anti_symmetric",
        "is_hermitian",
        "is_skew_hermitian",
        "is_unitary",
        "is_atomic",
        "is_sorted",
        "is_strictly_sorted",
        "is_unit_norm",
    }
)


def _normalize_value(key: str, value: Any) -> Any:
    if key in _BOOL_LIKE_KEYS:
        try:
            import numpy as np  # type: ignore

            if isinstance(value, np.bool_):
                return bool(value)
        except Exception:
            pass
    return value


def _validate_properties_compatibility(obj: Any, props: dict[str, Any]) -> None:
    """Minimal incompatibility checks (NOT truth validation).

    This rejects structurally impossible or internally contradictory asserted states.
    """

    shape = _get_shape(obj)
    square_n: int | None = None
    if shape is not None:
        r, c = shape
        if r == c:
            square_n = r

    # Shape constraints
    requires_square_true = (
        "is_unitary",
        "is_hermitian",
        "is_skew_hermitian",
        "is_symmetric",
        "is_anti_symmetric",
        "is_permutation",
        "is_upper_triangular",
        "is_lower_triangular",
        "is_atomic",
    )
    for k in requires_square_true:
        if props.get(k) is True and square_n is None and shape is not None:
            raise ValueError(f"{k}=True requires a square matrix")

    # Cached-derived shape constraints
    for k in ("trace", "determinant"):
        if k in props and square_n is None and shape is not None:
            raise ValueError(f"{k} is only defined for square matrices")

    # Internal contradictions / implication conflicts
    if props.get("is_identity") is True:
        # Identity is allowed rectangular, but it cannot also be asserted as zero for non-empty shapes.
        if props.get("is_zero") is True and shape is not None and (shape[0] * shape[1]) != 0:
            raise ValueError("is_identity=True contradicts is_zero=True")

        if "is_permutation" in props and props.get("is_permutation") is False:
            raise ValueError("is_identity=True implies is_permutation=True")
        if "is_diagonal" in props and props.get("is_diagonal") is False:
            raise ValueError("is_identity=True implies is_diagonal=True")
        if "is_upper_triangular" in props and props.get("is_upper_triangular") is False:
            raise ValueError("is_identity=True implies is_upper_triangular=True")
        if "is_lower_triangular" in props and props.get("is_lower_triangular") is False:
            raise ValueError("is_identity=True implies is_lower_triangular=True")

        if "has_unit_diagonal" in props and props.get("has_unit_diagonal") is False:
            raise ValueError("is_identity=True implies has_unit_diagonal=True")

        if "diagonal_value" in props and _as_number(props.get("diagonal_value")) != 1:
            raise ValueError("is_identity=True contradicts diagonal_value != 1")
        if props.get("has_zero_diagonal") is True:
            raise ValueError("is_identity=True contradicts has_zero_diagonal=True")

    if props.get("is_diagonal") is True:
        if "is_upper_triangular" in props and props.get("is_upper_triangular") is False:
            raise ValueError("is_diagonal=True implies is_upper_triangular=True")
        if "is_lower_triangular" in props and props.get("is_lower_triangular") is False:
            raise ValueError("is_diagonal=True implies is_lower_triangular=True")

    if props.get("has_unit_diagonal") is True:
        if props.get("has_zero_diagonal") is True:
            raise ValueError("has_unit_diagonal=True contradicts has_zero_diagonal=True")
        if "diagonal_value" in props and _as_number(props.get("diagonal_value")) != 1:
            raise ValueError("has_unit_diagonal=True contradicts diagonal_value != 1")

    if props.get("has_zero_diagonal") is True:
        if "diagonal_value" in props and _as_number(props.get("diagonal_value")) != 0:
            raise ValueError("has_zero_diagonal=True contradicts diagonal_value != 0")

    # Mutually-exclusive symmetry family assertions unless the user also asserts is_zero.
    if props.get("is_symmetric") is True and props.get("is_anti_symmetric") is True and props.get("is_zero") is not True:
        raise ValueError("is_symmetric=True and is_anti_symmetric=True requires is_zero=True")
    if props.get("is_hermitian") is True and props.get("is_skew_hermitian") is True and props.get("is_zero") is not True:
        raise ValueError("is_hermitian=True and is_skew_hermitian=True requires is_zero=True")

    # Triangular both-ways implies diagonal behavior; reject explicit negation.
    if props.get("is_upper_triangular") is True and props.get("is_lower_triangular") is True and props.get("is_diagonal") is False:
        raise ValueError("is_upper_triangular=True and is_lower_triangular=True contradicts is_diagonal=False")

    # Ordering constraints (vectors)
    if props.get("is_strictly_sorted") is True and props.get("is_sorted") is False:
        raise ValueError("is_strictly_sorted=True implies is_sorted=True")


class _PropertiesProxy(MutableMapping[str, Any]):
    __slots__ = ("_obj", "_store")

    def __init__(self, obj: Any, store: dict[str, Any]):
        self._obj = obj
        self._store = store

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError("properties keys must be strings")

        key = str(key)
        value = _normalize_value(key, value)

        prev = self._store.get(key, _MISSING)
        try:
            if value is None:
                self._store.pop(key, None)
            else:
                self._store[key] = value
            _validate_properties_compatibility(self._obj, self._store)
        except Exception:
            # revert
            if prev is _MISSING:
                self._store.pop(key, None)
            else:
                self._store[key] = prev
            raise

    def __delitem__(self, key: str) -> None:
        prev = self._store.get(key, _MISSING)
        if prev is _MISSING:
            raise KeyError(key)
        del self._store[key]
        try:
            _validate_properties_compatibility(self._obj, self._store)
        except Exception:
            self._store[key] = prev
            raise

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)


class _Missing:
    pass


_MISSING = _Missing()


def _ensure_proxy(obj: Any) -> MutableMapping[str, Any]:
    store = _ensure_store(obj)
    try:
        proxy = getattr(obj, "_pycauset_properties_proxy", None)
        if isinstance(proxy, _PropertiesProxy) and getattr(proxy, "_store", None) is store:
            return proxy
    except Exception:
        proxy = None

    proxy = _PropertiesProxy(obj, store)
    try:
        setattr(obj, "_pycauset_properties_proxy", proxy)
    except Exception:
        pass
    return proxy


def get_properties(obj: Any) -> dict[str, Any]:
    """Return the per-object properties mapping (created on demand)."""
    # Expose a validating mapping (Phase A) so incompatibilities fail immediately.
    return _ensure_proxy(obj)  # type: ignore[return-value]


def set_properties(obj: Any, mapping: Any) -> None:
    if mapping is None:
        mapping = {}
    from collections.abc import Mapping

    if not isinstance(mapping, Mapping):
        raise TypeError("properties must be a mapping")

    cleaned: dict[str, Any] = {}
    for k, v in mapping.items():
        if not isinstance(k, str):
            raise TypeError("properties keys must be strings")
        if v is None:
            # Tri-state booleans use presence semantics; None/unset is represented by missing.
            continue
        cleaned[k] = _normalize_value(k, v)

    _validate_properties_compatibility(obj, cleaned)

    try:
        setattr(obj, _PROPERTIES_ATTR, cleaned)
    except Exception:
        pass


def _bump_payload_epoch(obj: Any) -> None:
    """Best-effort in-memory epoch for payload mutation.

    This is intentionally not persisted yet; it exists to support cheap runtime
    invalidation decisions without scanning.
    """

    try:
        prev = getattr(obj, "_pycauset_payload_epoch", 0)
        setattr(obj, "_pycauset_payload_epoch", int(prev) + 1)
    except Exception:
        pass


def _invalidate_cached_derived(obj: Any) -> None:
    props = _get_store(obj)
    for k in _CACHED_DERIVED_KEYS:
        props.pop(k, None)

    # Large derived-cache (inverse) lives outside obj.properties.
    try:
        if hasattr(obj, "_cached_inverse"):
            delattr(obj, "_cached_inverse")
    except Exception:
        pass


def _post_payload_mutation(obj: Any) -> None:
    _bump_payload_epoch(obj)
    _invalidate_cached_derived(obj)


def _replace_properties_store(obj: Any, new_mapping: dict[str, Any]) -> None:
    store = _get_store(obj)
    store.clear()
    store.update(new_mapping)
    _validate_properties_compatibility(obj, store)


def _post_view_transform_inplace(obj: Any, *, op: str) -> None:
    store = _get_store(obj)
    is_real = _is_real_dtype(obj)
    new_props = _propagate_mapping(store, op=op, is_real=is_real)
    _replace_properties_store(obj, new_props)


def _apply_scalar_ratio_inplace(obj: Any, *, ratio: Any) -> None:
    store = _get_store(obj)
    ratio = _as_number(ratio)

    # Cached-derived values (including scalar=0, when the cache key is present).
    if "trace" in store:
        store["trace"] = _as_number(store["trace"]) * ratio
    if "sum" in store:
        store["sum"] = _as_number(store["sum"]) * ratio
    if "determinant" in store:
        n = _get_square_dim(obj)
        if n is None:
            store.pop("determinant", None)
        else:
            store["determinant"] = _as_number(store["determinant"]) * (ratio**n)
    if "norm" in store:
        store["norm"] = _as_number(store["norm"]) * abs(ratio)
    if "rank" in store:
        # scaling by nonzero preserves rank; scaling by zero yields rank 0.
        if ratio == 0:
            store["rank"] = 0

    # Scalar affects constant-diagonal metadata deterministically.
    if "diagonal_value" in store:
        store["diagonal_value"] = _as_number(store["diagonal_value"]) * ratio

    # Diagonal shorthands: preserve True only when deterministic.
    if store.get("has_unit_diagonal") is True and ratio != 1:
        store.pop("has_unit_diagonal", None)

    # Gospel properties affected by scaling
    if store.get("is_identity") is True and ratio != 1:
        store["is_identity"] = False
    if store.get("is_permutation") is True and ratio != 1:
        store["is_permutation"] = False
    if store.get("is_unitary") is True:
        try:
            if abs(ratio) != 1:
                store.pop("is_unitary", None)
        except Exception:
            store.pop("is_unitary", None)

    _validate_properties_compatibility(obj, store)


def _apply_effect_summary_inplace(obj: Any, summary: dict[str, Any]) -> None:
    """Apply a constant-size effect summary to properties without scanning payload.

    Keys (all optional, conservative-only when unknown):
    - off_diagonal_nonzero: True if any nonzero off-diagonal entry was written.
    - off_diagonal_side: "upper", "lower", or "both" (ignored otherwise).
    - diagonal_written: True if any diagonal entry was written.
    - diagonal_value: last written diagonal value (used when diagonal_written).
    - known_all_zero: True when the resulting payload is known to be all zeros.
    - set_identity: True when the resulting payload is known to be identity.
    """

    store = _get_store(obj)

    try:
        if summary.get("set_identity"):
            store["is_identity"] = True
            store["is_diagonal"] = True
            store["is_upper_triangular"] = True
            store["is_lower_triangular"] = True
            store["has_unit_diagonal"] = True
            store["diagonal_value"] = 1
            store.pop("is_zero", None)
            store.pop("has_zero_diagonal", None)

        if summary.get("known_all_zero"):
            store["is_zero"] = True
            if store.get("is_identity") is True:
                store["is_identity"] = False
            store.pop("has_unit_diagonal", None)
            store.pop("diagonal_value", None)
            store.pop("has_zero_diagonal", None)

        if summary.get("off_diagonal_nonzero"):
            store["is_zero"] = False
            if store.get("is_diagonal") is True:
                store["is_diagonal"] = False
            if store.get("is_identity") is True:
                store["is_identity"] = False

            side = summary.get("off_diagonal_side")
            # A nonzero write above the diagonal invalidates lower-triangular;
            # a nonzero write below the diagonal invalidates upper-triangular.
            if side in ("upper", "both") and store.get("is_lower_triangular") is True:
                store["is_lower_triangular"] = False
            if side in ("lower", "both") and store.get("is_upper_triangular") is True:
                store["is_upper_triangular"] = False

            if store.get("has_unit_diagonal") is True:
                store.pop("has_unit_diagonal", None)
            store.pop("diagonal_value", None)

        if summary.get("diagonal_written"):
            if "diagonal_value" in summary:
                dv = summary.get("diagonal_value")
                dv_num = _as_number(dv)

                if store.get("is_zero") is True:
                    try:
                        if dv_num != 0:
                            store["is_zero"] = False
                    except Exception:
                        store["is_zero"] = False

                if store.get("has_zero_diagonal") is True:
                    try:
                        if dv_num != 0:
                            store["has_zero_diagonal"] = False
                    except Exception:
                        store["has_zero_diagonal"] = False

                if store.get("has_unit_diagonal") is True:
                    try:
                        if dv_num != 1:
                            store["has_unit_diagonal"] = False
                    except Exception:
                        store["has_unit_diagonal"] = False

                if store.get("is_identity") is True:
                    try:
                        if dv_num != 1:
                            store["is_identity"] = False
                    except Exception:
                        store["is_identity"] = False

                if "diagonal_value" in store:
                    try:
                        if _as_number(store["diagonal_value"]) != dv_num:
                            store.pop("diagonal_value", None)
                    except Exception:
                        store.pop("diagonal_value", None)

    except Exception:
        pass

    try:
        _validate_properties_compatibility(obj, store)
    except Exception:
        pass


def apply_properties_patches(*, classes: list[Any]) -> None:
    for cls in classes:
        if cls is None:
            continue

        # Don't re-install if already present.
        if isinstance(getattr(cls, "properties", None), property):
            continue

        def _get(self: Any) -> dict[str, Any]:
            return get_properties(self)

        def _set(self: Any, value: Any) -> None:
            set_properties(self, value)

        try:
            cls.properties = property(_get, _set)  # type: ignore[attr-defined]
        except Exception:
            # Best-effort: some native types may be immutable.
            continue


def _is_real_dtype(obj: Any) -> bool:
    name = type(obj).__name__
    return "Complex" not in name and "complex" not in name


def _conj_value(value: Any) -> Any:
    try:
        # complex supports .conjugate(); floats/ints return self
        return value.conjugate()  # type: ignore[attr-defined]
    except Exception:
        return value


def _propagate_mapping(mapping: dict[str, Any], *, op: str, is_real: bool) -> dict[str, Any]:
    out = dict(mapping)

    def _unset(key: str) -> None:
        out.pop(key, None)

    if op == "transpose":
        up = out.get("is_upper_triangular", None)
        low = out.get("is_lower_triangular", None)
        if "is_upper_triangular" in out or "is_lower_triangular" in out:
            if "is_upper_triangular" in out:
                out["is_lower_triangular"] = up
            else:
                out.pop("is_lower_triangular", None)
            if "is_lower_triangular" in out:
                out["is_upper_triangular"] = low
            else:
                out.pop("is_upper_triangular", None)

        if not is_real:
            _unset("is_hermitian")
            _unset("is_skew_hermitian")

        _unset("is_unitary")

        # Cached-derived values
        # trace, determinant, eigenvalues: preserved under transpose when present.
        return out

    if op == "conj":
        if "diagonal_value" in out:
            out["diagonal_value"] = _conj_value(out["diagonal_value"])

        if not is_real:
            _unset("is_hermitian")
            _unset("is_skew_hermitian")

        _unset("is_unitary")

        if "trace" in out:
            out["trace"] = _conj_value(out["trace"])
        if "determinant" in out:
            out["determinant"] = _conj_value(out["determinant"])
        if "sum" in out:
            out["sum"] = _conj_value(out["sum"])
        if "eigenvalues" in out and isinstance(out.get("eigenvalues"), list):
            out["eigenvalues"] = [_conj_value(v) for v in out["eigenvalues"]]

        return out

    if op == "adjoint":
        # transpose + conjugation, with exceptions
        out = _propagate_mapping(out, op="transpose", is_real=is_real)
        out = _propagate_mapping(out, op="conj", is_real=is_real)

        # exceptions: preserve these keys (by original value)
        for k in ("is_unitary", "is_hermitian", "is_skew_hermitian"):
            if k in mapping:
                out[k] = mapping[k]
            else:
                out.pop(k, None)
        return out

    return out


def _copy_and_propagate_properties(parent: Any, child: Any, *, op: str) -> None:
    parent_props = get_properties(parent)
    is_real = _is_real_dtype(parent)
    new_props = _propagate_mapping(parent_props, op=op, is_real=is_real)
    set_properties(child, new_props)


def apply_properties_view_patches(
    *,
    classes: list[Any],
    track_matrix: Any,
    mark_temporary_if_auto: Any,
) -> None:
    """Patch metadata-only view constructors to propagate obj.properties.

    This is a best-effort Python-level shim; it does not scan payload data.
    """

    for cls in classes:
        if cls is None:
            continue

        # transpose()
        orig_transpose = getattr(cls, "transpose", None)
        if callable(orig_transpose) and not getattr(orig_transpose, "_pycauset_props_wrapped", False):

            def _make_transpose_wrapper(orig: Any) -> Any:
                def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    out = orig(self, *args, **kwargs)
                    try:
                        _copy_and_propagate_properties(self, out, op="transpose")
                    except Exception:
                        pass
                    try:
                        track_matrix(out)
                        mark_temporary_if_auto(out)
                    except Exception:
                        pass
                    return out

                _wrapped._pycauset_props_wrapped = True  # type: ignore[attr-defined]
                return _wrapped

            try:
                setattr(cls, "transpose", _make_transpose_wrapper(orig_transpose))
            except Exception:
                pass

        # Ensure .T uses patched transpose.
        try:
            cls.T = property(lambda self: self.transpose())  # type: ignore[attr-defined]
        except Exception:
            pass

        # conj()
        orig_conj = getattr(cls, "conj", None)
        if callable(orig_conj) and not getattr(orig_conj, "_pycauset_props_wrapped", False):

            def _make_conj_wrapper(orig: Any) -> Any:
                def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    out = orig(self, *args, **kwargs)
                    try:
                        _copy_and_propagate_properties(self, out, op="conj")
                    except Exception:
                        pass
                    try:
                        track_matrix(out)
                        mark_temporary_if_auto(out)
                    except Exception:
                        pass
                    return out

                _wrapped._pycauset_props_wrapped = True  # type: ignore[attr-defined]
                return _wrapped

            try:
                setattr(cls, "conj", _make_conj_wrapper(orig_conj))
            except Exception:
                pass

        # .H (adjoint)
        orig_H = getattr(cls, "H", None)
        if orig_H is not None and hasattr(orig_H, "__get__"):
            try:
                # Avoid re-wrapping if already a Python property we installed.
                if not (isinstance(orig_H, property) and getattr(orig_H.fget, "_pycauset_props_wrapped", False)):

                    def _make_H_getter(descr: Any) -> Any:
                        def _getter(self: Any) -> Any:
                            out = descr.__get__(self, cls)
                            try:
                                _copy_and_propagate_properties(self, out, op="adjoint")
                            except Exception:
                                pass
                            try:
                                track_matrix(out)
                                mark_temporary_if_auto(out)
                            except Exception:
                                pass
                            return out

                        _getter._pycauset_props_wrapped = True  # type: ignore[attr-defined]
                        return _getter

                    cls.H = property(_make_H_getter(orig_H))  # type: ignore[attr-defined]
            except Exception:
                pass


def _get_square_dim(obj: Any) -> int | None:
    try:
        r = int(obj.rows())
        c = int(obj.cols())
        if r == c:
            return r
        return None
    except Exception:
        try:
            shape = getattr(obj, "shape", None)
            if isinstance(shape, tuple) and len(shape) == 2 and int(shape[0]) == int(shape[1]):
                return int(shape[0])
        except Exception:
            pass
    return None


def _as_number(value: Any) -> Any:
    # Accept ints/floats/complex and numpy scalars.
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _is_scalar_value(value: Any) -> bool:
    if isinstance(value, (bool, int, float, complex)):
        return True
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return True
    except Exception:
        pass
    return False


def effective_structure_from_properties(props: Any) -> str:
    """Compute the effective structure category from properties (priority tree).

    This must be $O(1)$ and must not scan payload.

    Returns one of:
    - "zero"
    - "identity"
    - "diagonal"
    - "upper_triangular"
    - "lower_triangular"
    - "general"
    """

    try:
        is_zero = props.get("is_zero") is True
        is_identity = props.get("is_identity") is True
        is_diagonal = props.get("is_diagonal") is True
        is_upper = props.get("is_upper_triangular") is True
        is_lower = props.get("is_lower_triangular") is True
    except Exception:
        return "general"

    if is_zero:
        return "zero"
    if is_identity:
        return "identity"
    if is_diagonal or (is_upper and is_lower):
        return "diagonal"
    if is_upper:
        return "upper_triangular"
    if is_lower:
        return "lower_triangular"
    return "general"


def effective_structure(obj: Any) -> str:
    """Compute effective structure category for an object."""
    return effective_structure_from_properties(get_properties(obj))


def apply_properties_arithmetic_patches(*, classes: list[Any]) -> None:
    """Patch arithmetic operators to propagate cached-derived values where safe.

    This is Phase A/D glue: scalar multiply and add/sub cached-derived propagation.
    """

    for cls in classes:
        if cls is None:
            continue

        # __mul__ / __rmul__ (scalar multiply)
        for op_name, scalar_side in (("__mul__", "right"), ("__rmul__", "left")):
            orig = getattr(cls, op_name, None)
            if not callable(orig) or getattr(orig, "_pycauset_props_arith_wrapped", False):
                continue

            def _make_mul_wrapper(orig_method: Any, *, side: str) -> Any:
                def _wrapped(self: Any, other: Any, *args: Any, **kwargs: Any) -> Any:
                    out = orig_method(self, other, *args, **kwargs)

                    # Only handle scalar multiply; leave matrix-matrix and other overloads alone.
                    scalar = other if side == "right" else other
                    if not _is_scalar_value(scalar):
                        return out

                    try:
                        parent = self
                        parent_store = _get_store(parent)
                        new_props = dict(parent_store)
                        _replace_properties_store(out, new_props)
                        _apply_scalar_ratio_inplace(out, ratio=_as_number(scalar))
                    except Exception:
                        pass
                    return out

                _wrapped._pycauset_props_arith_wrapped = True  # type: ignore[attr-defined]
                _wrapped._pycauset_props_arith_op = op_name  # type: ignore[attr-defined]
                return _wrapped

            try:
                setattr(cls, op_name, _make_mul_wrapper(orig, side=scalar_side))
            except Exception:
                pass

        # __add__ / __sub__ (cached-derived propagation)
        for op_name in ("__add__", "__sub__"):
            orig = getattr(cls, op_name, None)
            if not callable(orig) or getattr(orig, "_pycauset_props_arith_wrapped", False):
                continue

            def _make_addsub_wrapper(orig_method: Any, *, op: str) -> Any:
                def _wrapped(self: Any, other: Any, *args: Any, **kwargs: Any) -> Any:
                    out = orig_method(self, other, *args, **kwargs)

                    try:
                        left = _get_store(self)
                        right = _get_store(other)
                        new_props: dict[str, Any] = {}

                        # Cached-derived: trace can be propagated if both known.
                        if "trace" in left and "trace" in right:
                            if op == "add":
                                new_props["trace"] = _as_number(left["trace"]) + _as_number(right["trace"])
                            else:
                                new_props["trace"] = _as_number(left["trace"]) - _as_number(right["trace"])

                        # Cached-derived: sum can be propagated if both known.
                        if "sum" in left and "sum" in right:
                            if op == "add":
                                new_props["sum"] = _as_number(left["sum"]) + _as_number(right["sum"])
                            else:
                                new_props["sum"] = _as_number(left["sum"]) - _as_number(right["sum"])

                        # determinant/rank/norm: conservative unset (per table).
                        _replace_properties_store(out, new_props)
                    except Exception:
                        pass
                    return out

                _wrapped._pycauset_props_arith_wrapped = True  # type: ignore[attr-defined]
                _wrapped._pycauset_props_arith_op = op_name  # type: ignore[attr-defined]
                return _wrapped

            try:
                setattr(cls, op_name, _make_addsub_wrapper(orig, op=("add" if op_name == "__add__" else "sub")))
            except Exception:
                pass


def apply_properties_operator_patches(*, classes: list[Any]) -> None:
    """Patch scalar-returning operators to consult `obj.properties` first.

    Phase E minimal: trace() and determinant().
    """

    for cls in classes:
        if cls is None:
            continue

        # trace()
        orig_trace = getattr(cls, "trace", None)
        if callable(orig_trace) and not getattr(orig_trace, "_pycauset_props_wrapped", False):

            def _make_trace_wrapper(orig: Any) -> Any:
                def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    props = get_properties(self)
                    if "trace" in props:
                        return _as_number(props["trace"])

                    n = _get_square_dim(self)
                    if props.get("is_zero") is True:
                        val = 0
                        props["trace"] = val
                        return val

                    if props.get("is_identity") is True and n is not None:
                        val = n
                        props["trace"] = val
                        return val

                    # Constant diagonal shortcut when user asserts diagonal.
                    is_diag = props.get("is_diagonal") is True or (
                        props.get("is_upper_triangular") is True and props.get("is_lower_triangular") is True
                    )
                    if is_diag and n is not None and "diagonal_value" in props:
                        dv = _as_number(props["diagonal_value"])
                        val = dv * n
                        props["trace"] = val
                        return val

                    val = orig(self, *args, **kwargs)
                    try:
                        props["trace"] = _as_number(val)
                    except Exception:
                        pass
                    return val

                _wrapped._pycauset_props_wrapped = True  # type: ignore[attr-defined]
                return _wrapped

            try:
                setattr(cls, "trace", _make_trace_wrapper(orig_trace))
            except Exception:
                pass


def apply_properties_mutation_patches(*, classes: list[Any]) -> None:
    """Patch known in-place mutation methods to invalidate cached-derived values."""

    payload_mutation_method_names = (
        "set",
        "fill",
        "set_identity",
    )

    view_mutation_method_names = (
        "set_transposed",
        "set_conjugated",
        "set_scalar",
        "set_temporary",
    )

    for cls in classes:
        if cls is None:
            continue

        for name in payload_mutation_method_names:
            orig = getattr(cls, name, None)
            if not callable(orig):
                continue
            if getattr(orig, "_pycauset_props_mutation_wrapped", False):
                continue

            def _make_wrapper(method_name: str, orig_method: Any) -> Any:
                def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    out = orig_method(self, *args, **kwargs)
                    summary: dict[str, Any] = {"payload_mutated": True}

                    # Effect-summary style updates: deterministically clear/adjust
                    # gospel properties based only on mutation parameters.
                    try:
                        if method_name == "set" and len(args) >= 3:
                            i = int(args[0])
                            j = int(args[1])
                            val = args[2]

                            def _is_nonzero(x: Any) -> bool:
                                try:
                                    return x != 0
                                except Exception:
                                    return True

                            if i == j:
                                summary["diagonal_written"] = True
                                summary["diagonal_value"] = val
                            else:
                                if _is_nonzero(val):
                                    summary["off_diagonal_nonzero"] = True
                                    summary["off_diagonal_side"] = "upper" if j > i else "lower"

                        elif method_name == "fill" and len(args) >= 1:
                            val = args[0]

                            def _is_zero_fill(x: Any) -> bool:
                                try:
                                    return x == 0
                                except Exception:
                                    return False

                            zero_fill = _is_zero_fill(val)
                            summary["known_all_zero"] = zero_fill
                            summary["diagonal_written"] = True
                            summary["diagonal_value"] = val
                            if not zero_fill:
                                summary["off_diagonal_nonzero"] = True
                                summary["off_diagonal_side"] = "both"

                        elif method_name == "set_identity":
                            summary["set_identity"] = True

                        _apply_effect_summary_inplace(self, summary)
                    except Exception:
                        pass

                    try:
                        _post_payload_mutation(self)
                    except Exception:
                        pass
                    return out

                _wrapped._pycauset_props_mutation_wrapped = True  # type: ignore[attr-defined]
                _wrapped._pycauset_props_mutation_method = method_name  # type: ignore[attr-defined]
                return _wrapped

            try:
                setattr(cls, name, _make_wrapper(name, orig))
            except Exception:
                pass

        # View-state mutations (scalar / transpose / conjugation toggles).
        for name in view_mutation_method_names:
            orig = getattr(cls, name, None)
            if not callable(orig):
                continue
            if getattr(orig, "_pycauset_props_view_mutation_wrapped", False):
                continue

            def _make_view_wrapper(method_name: str, orig_method: Any) -> Any:
                def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    if method_name == "set_transposed":
                        before = None
                        try:
                            before = bool(self.is_transposed())
                        except Exception:
                            pass
                        out = orig_method(self, *args, **kwargs)
                        after = None
                        try:
                            after = bool(self.is_transposed())
                        except Exception:
                            pass
                        if before is not None and after is not None and before != after:
                            try:
                                _post_view_transform_inplace(self, op="transpose")
                            except Exception:
                                pass
                        return out

                    if method_name == "set_conjugated":
                        before = None
                        try:
                            before = bool(self.is_conjugated())
                        except Exception:
                            pass
                        out = orig_method(self, *args, **kwargs)
                        after = None
                        try:
                            after = bool(self.is_conjugated())
                        except Exception:
                            pass
                        if before is not None and after is not None and before != after:
                            try:
                                _post_view_transform_inplace(self, op="conj")
                            except Exception:
                                pass
                        return out

                    if method_name == "set_scalar":
                        old = None
                        try:
                            old = getattr(self, "scalar", 1.0)
                        except Exception:
                            pass
                        out = orig_method(self, *args, **kwargs)
                        new = None
                        try:
                            new = getattr(self, "scalar", 1.0)
                        except Exception:
                            pass

                        if old is None or new is None:
                            return out

                        try:
                            old_n = _as_number(old)
                            new_n = _as_number(new)
                            if old_n == 0:
                                _invalidate_cached_derived(self)
                            else:
                                _apply_scalar_ratio_inplace(self, ratio=(new_n / old_n))
                        except Exception:
                            # Conservative: if ratio math fails, clear cached-derived.
                            _invalidate_cached_derived(self)
                        return out

                    # set_temporary or unknown view-state mutation: conservative.
                    out = orig_method(self, *args, **kwargs)
                    try:
                        _invalidate_cached_derived(self)
                    except Exception:
                        pass
                    return out

                _wrapped._pycauset_props_view_mutation_wrapped = True  # type: ignore[attr-defined]
                _wrapped._pycauset_props_view_mutation_method = method_name  # type: ignore[attr-defined]
                return _wrapped

            try:
                setattr(cls, name, _make_view_wrapper(name, orig))
            except Exception:
                pass

        # determinant()
        orig_det = getattr(cls, "determinant", None)
        if callable(orig_det) and not getattr(orig_det, "_pycauset_props_wrapped", False):

            def _make_det_wrapper(orig: Any) -> Any:
                def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    props = get_properties(self)
                    if "determinant" in props:
                        return _as_number(props["determinant"])

                    n = _get_square_dim(self)
                    if props.get("is_zero") is True:
                        val = 0
                        props["determinant"] = val
                        return val

                    if props.get("is_identity") is True:
                        val = 1
                        props["determinant"] = val
                        return val

                    is_diag = props.get("is_diagonal") is True or (
                        props.get("is_upper_triangular") is True and props.get("is_lower_triangular") is True
                    )
                    if is_diag and n is not None and "diagonal_value" in props:
                        dv = _as_number(props["diagonal_value"])
                        val = dv**n
                        props["determinant"] = val
                        return val

                    val = orig(self, *args, **kwargs)
                    try:
                        props["determinant"] = _as_number(val)
                    except Exception:
                        pass
                    return val

                _wrapped._pycauset_props_wrapped = True  # type: ignore[attr-defined]
                return _wrapped

            try:
                setattr(cls, "determinant", _make_det_wrapper(orig_det))
            except Exception:
                pass
