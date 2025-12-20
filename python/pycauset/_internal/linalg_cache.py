from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .persistence import zip_member_data_offset


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

        backing = self.get_backing_file()
        if backing and backing.endswith(".pycauset") and os.path.exists(backing):
            try:
                with zipfile.ZipFile(backing, "r") as zf:
                    if "inverse.bin" in zf.namelist():
                        info = zf.getinfo("inverse.bin")
                        offset = zip_member_data_offset(backing, info)

                        try:
                            inv = FloatMatrix._from_storage(
                                self.rows(), self.cols(), backing, offset, 0, 1.0, False
                            )
                        except TypeError:
                            inv = FloatMatrix._from_storage(self.rows(), backing, offset, 0, 1.0, False)
                        self._cached_inverse = inv
                        return inv
            except Exception:
                pass

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

        if save:
            if backing and backing.endswith(".pycauset") and os.path.exists(backing):
                with zipfile.ZipFile(backing, "a") as zf:
                    inv_file = inv.get_backing_file()
                    if inv_file and inv_file != ":memory:" and os.path.exists(inv_file):
                        zf.write(inv_file, "inverse.bin")
                    else:
                        temp_inv = str(Path(backing).with_suffix(".inv.tmp"))
                        inv.copy_storage(temp_inv)
                        zf.write(temp_inv, "inverse.bin")
                        try:
                            os.unlink(temp_inv)
                        except OSError:
                            pass

        return inv

    return _inverse


def patch_inverse(*, FloatMatrix: Any, classes: list[Any]) -> None:
    inv_func = make_inverse(FloatMatrix)

    for cls in classes:
        if cls and hasattr(cls, "invert"):
            cls._invert_native = cls.invert
            cls.invert = inv_func
