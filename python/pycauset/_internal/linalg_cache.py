from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

_PYCAUSET_MAGIC = b"PYCAUSET"

from . import persistence as _persistence
from . import big_blob_cache as _big_blob_cache


def _is_new_container(backing: str) -> bool:
    try:
        with open(backing, "rb") as f:
            return f.read(len(_PYCAUSET_MAGIC)) == _PYCAUSET_MAGIC
    except OSError:
        return False


def patch_matrixbase_save(native: Any, save_func: Any) -> None:
    if hasattr(native, "MatrixBase"):
        native.MatrixBase.save = save_func
    if hasattr(native, "VectorBase"):
        native.VectorBase.save = save_func


def make_inverse(FloatMatrix: Any) -> Any:
    def _inverse(self: Any, save: bool = False) -> Any:
        """Compute or retrieve cached inverse."""
        if hasattr(self, "_cached_inverse"):
            return self._cached_inverse

        backing = None
        try:
            backing = self.get_backing_file()
        except Exception:
            backing = None

        if backing and isinstance(self, FloatMatrix) and backing.endswith(".pycauset") and os.path.exists(backing):
            if _is_new_container(backing):
                # If a persisted inverse object exists for this view state, load it.
                ref_exists = False
                try:
                    view_sig = _big_blob_cache.compute_view_signature(self)
                    ref_exists = isinstance(
                        _persistence.try_get_cached_big_blob_ref(
                            backing,
                            name="inverse",
                            view_signature=view_sig,
                        ),
                        dict,
                    )
                    inv = _big_blob_cache.try_load_cached_matrix(
                        backing,
                        name="inverse",
                        view_signature=view_sig,
                        MatrixClass=FloatMatrix,
                    )
                except Exception:
                    inv = None

                if inv is not None:
                    self._cached_inverse = inv
                    return inv

                # Cache reference exists but the referenced object is missing/unreadable.
                # Policy: warn (handled in big_blob_cache) and do NOT implicitly recompute.
                # Users can explicitly rebuild the cache by calling invert(save=True).
                if ref_exists and not save:
                    raise FileNotFoundError(
                        "cached inverse object is missing or unreadable; "
                        "no implicit recompute is performed. "
                        "Call invert(save=True) to rebuild the cached inverse."
                    )

                # Correctness-first: native invert currently produces invalid results for
                # matrices backed by the new container (likely ignoring payload offsets).
                # Until native inversion is made container-aware, force a NumPy inverse.
                inv_np = np.linalg.inv(np.asarray(self))
                inv = FloatMatrix(inv_np)
                self._cached_inverse = inv

                if save:
                    try:
                        _big_blob_cache.persist_cached_object(
                            backing,
                            name="inverse",
                            obj=inv,
                            view_signature=view_sig,
                        )
                    except Exception:
                        # Best-effort cache: computation result still returned.
                        pass

                return inv

        if hasattr(self, "_invert_native"):
            try:
                inv = self._invert_native()
            except Exception as exc:
                # Correctness-first fallback: if native inversion is unavailable/buggy,
                # attempt a NumPy CPU inverse. Preserve original error if NumPy fails
                # (e.g., truly singular matrices).
                try:
                    inv_np = np.linalg.inv(np.asarray(self))
                    inv = FloatMatrix(inv_np)
                except Exception:
                    raise exc
        else:
            raise NotImplementedError("Native invert not found")

        self._cached_inverse = inv

        # If requested, persist inverse for container-backed matrices.
        if save and backing and backing.endswith(".pycauset") and os.path.exists(backing) and _is_new_container(backing):
            try:
                view_sig = _big_blob_cache.compute_view_signature(self)
                _big_blob_cache.persist_cached_object(
                    backing,
                    name="inverse",
                    obj=inv,
                    view_signature=view_sig,
                )
            except Exception:
                pass

        return inv

    return _inverse


def patch_inverse(*, FloatMatrix: Any, classes: list[Any]) -> None:
    inv_func = make_inverse(FloatMatrix)

    for cls in classes:
        if cls and hasattr(cls, "invert"):
            cls._invert_native = cls.invert
            cls.invert = inv_func
