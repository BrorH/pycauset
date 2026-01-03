from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple


_METHODS_RETURNING_MATRIX = [
    "multiply",
    "invert",
    "__invert__",
    "__add__",
    "__sub__",
    "__mul__",
    "__rmul__",
    "transpose",
    "conj",
]
def patch_matrix_methods(
    cls: Any,
    *,
    maybe_register_result: Callable[[Any], None],
) -> None:
    for name in _METHODS_RETURNING_MATRIX:
        if not hasattr(cls, name):
            continue
        original = getattr(cls, name)

        def make_wrapper(orig: Any) -> Any:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = orig(self, *args, **kwargs)
                maybe_register_result(result)
                return result

            return wrapper

        setattr(cls, name, make_wrapper(original))


def patch_matrix_class(
    cls: Any,
    *,
    track_matrix: Callable[[Any], None],
    mark_temporary_if_auto: Callable[[Any], None],
) -> None:
    original_init = cls.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        mutable_args = list(args)
        if len(mutable_args) > 0 and isinstance(mutable_args[0], float):
            mutable_args[0] = int(mutable_args[0])
        elif "N" in kwargs and isinstance(kwargs["N"], float):
            kwargs["N"] = int(kwargs["N"])

        original_init(self, *mutable_args, **kwargs)
        track_matrix(self)
        mark_temporary_if_auto(self)

    cls.__init__ = _patched_init

    def _maybe_register_result(result: Any) -> None:
        if result is None:
            return
        try:
            if not hasattr(result, "get_backing_file"):
                return
        except Exception:
            return
        track_matrix(result)
        mark_temporary_if_auto(result)

    patch_matrix_methods(cls, maybe_register_result=_maybe_register_result)

    original_random = getattr(cls, "random", None)
    if callable(original_random):
        def _patched_random(*args: Any, **kwargs: Any) -> Any:
            result = original_random(*args, **kwargs)
            _maybe_register_result(result)
            return result

        cls.random = staticmethod(_patched_random)


def apply_native_storage_patches(
    *,
    matrix_classes: list[tuple[Any | None, str]],
    vector_classes: list[tuple[Any | None, str]],
    track_matrix: Callable[[Any], None],
    mark_temporary_if_auto: Callable[[Any], None],
) -> None:
    for cls, _target_arg in matrix_classes:
        if cls:
            patch_matrix_class(
                cls,
                track_matrix=track_matrix,
                mark_temporary_if_auto=mark_temporary_if_auto,
            )

    for cls, _target_arg in vector_classes:
        if cls:
            patch_matrix_class(
                cls,
                track_matrix=track_matrix,
                mark_temporary_if_auto=mark_temporary_if_auto,
            )
