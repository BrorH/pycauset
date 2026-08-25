import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

_STORAGE_TMP = tempfile.TemporaryDirectory()
import pycauset as pc

pc.set_backing_dir(_STORAGE_TMP.name)


# Internal native machinery that must not be star-exported (still reachable as
# pycauset.<name> via __getattr__).
_INTERNAL = frozenset(
    {
        "LazyMatrix",
        "lazy_add",
        "lazy_cos",
        "lazy_mul_scalar",
        "lazy_sin",
        "lazy_sub",
        "MemoryGovernor",
        "IOAccelerator",
        "OpContract",
        "OpRegistry",
        "get_storage_root",
        "set_storage_root",
        "make_coordinates",
        "sprinkle",
    }
)

# A representative sample of the public surface that must remain star-exported.
_PUBLIC = frozenset(
    {
        "matmul",
        "solve",
        "svd",
        "eigh",
        "identity",
        "symmetric",
        "antisymmetric",
        "diagonal",
        "FloatMatrix",
        "SymmetricMatrix",
        "AntiSymmetricMatrix",
        "DiagonalMatrix",
        "CausalSet",
        "TriangularMatrix",
        "to_numpy",
        "load",
        "save",
        "dot",
    }
)


class TestPublicApiSurface(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        _STORAGE_TMP.cleanup()

    def test_internal_symbols_not_in_all(self):
        for name in _INTERNAL:
            self.assertNotIn(name, pc.__all__, f"{name} should not be public API")

    def test_internal_symbols_still_reachable(self):
        for name in _INTERNAL:
            self.assertTrue(hasattr(pc, name), f"{name} should remain reachable")

    def test_public_symbols_in_all(self):
        for name in _PUBLIC:
            self.assertIn(name, pc.__all__, f"{name} should be public API")

    def test_star_import_excludes_internals(self):
        ns = {}
        exec("from pycauset import *", ns)
        for name in _INTERNAL:
            self.assertNotIn(name, ns, f"star import leaked internal {name}")
        self.assertIn("matmul", ns)


if __name__ == "__main__":
    unittest.main()
