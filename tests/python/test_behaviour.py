import atexit
import gc
import os
import shutil
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
atexit.register(_STORAGE_TMP.cleanup)
os.environ["PYCAUSET_STORAGE_DIR"] = _STORAGE_TMP.name

import pycauset


class CausalMatrixBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.storage_dir = Path(os.environ["PYCAUSET_STORAGE_DIR"])
        self._cleanup_storage_dir()

    def tearDown(self):
        gc.collect()
        self._cleanup_storage_dir()
        pycauset.save = False
        pycauset.seed = None

    def _cleanup_storage_dir(self):
        for child in self.storage_dir.glob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
            else:
                shutil.rmtree(child)

    def test_getitem_returns_integer_bits(self):
        alpha = pycauset.causalmatrix(3, populate=False)
        alpha[0, 1] = True
        value = alpha[0, 1]
        self.assertEqual(value, 1)
        self.assertIs(type(value), int)
        expected_path = self.storage_dir / "alpha.pycauset"
        self.assertTrue(expected_path.exists())
        alpha = None

    def test_pseudobool_assignments(self):
        mat = pycauset.causalmatrix(3, populate=False)
        mat[0, 1] = 1
        mat[0, 2] = 1.0
        self.assertEqual(mat[0, 1], 1)
        self.assertEqual(mat[0, 2], 1)
        with self.assertRaises(TypeError):
            mat[1, 2] = 2
        with self.assertRaises(TypeError):
            mat.set(0, 2, 3.14)
        mat = None

    def test_simple_name_resolves_inside_storage_dir(self):
        beta = pycauset.causalmatrix(3, "custom_file", populate=False)
        expected = self.storage_dir / "custom_file.pycauset"
        self.assertTrue(expected.exists())
        beta = None

    def test_absolute_path_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "explicit.bin"
            gamma = pycauset.causalmatrix(3, raw_path, populate=False)
            final_path = raw_path.with_suffix(raw_path.suffix + ".pycauset")
            self.assertTrue(final_path.exists())
            gamma = None

    def test_matmul_creates_new_integer_matrix(self):
        mat = pycauset.causalmatrix(3, populate=False)
        mat[0, 1] = 1
        mat[1, 2] = 1
        product = pycauset.matmul(mat, mat)
        self.assertIsInstance(product, pycauset.IntegerMatrix)
        self.assertEqual(product[0, 2], 1)
        mat = None
        product = None

    def test_elementwise_mul_is_componentwise(self):
        lhs = pycauset.causalmatrix(3, populate=False)
        rhs = pycauset.causalmatrix(3, populate=False)
        lhs[0, 1] = 1
        lhs[0, 2] = 1
        rhs[0, 1] = 1
        rhs[1, 2] = 1
        product = lhs * rhs
        self.assertIsInstance(product, pycauset.causalmatrix)
        self.assertEqual(product[0, 1], 1)
        self.assertEqual(product[0, 2], 0)
        self.assertEqual(product[1, 2], 0)

    def test_auto_backing_removed_when_save_disabled(self):
        pycauset.save = False
        mat = pycauset.causalmatrix(3, populate=False)
        mat_name = self.storage_dir / "mat.pycauset"
        self.assertTrue(mat_name.exists())
        mat = None
        gc.collect()
        pycauset._STORAGE_REGISTRY._flush(pycauset.save)  # type: ignore[attr-defined]
        self.assertFalse(mat_name.exists())

    def test_saved_backing_removed_on_future_false_run(self):
        pycauset.save = True
        mat = pycauset.causalmatrix(3, populate=False)
        saved_path = self.storage_dir / "mat.pycauset"
        self.assertTrue(saved_path.exists())
        mat = None
        gc.collect()
        pycauset._STORAGE_REGISTRY._flush(pycauset.save)  # type: ignore[attr-defined]
        self.assertTrue(saved_path.exists())

        pycauset.save = False
        pycauset._STORAGE_REGISTRY._flush(pycauset.save)  # type: ignore[attr-defined]
        self.assertFalse(saved_path.exists())

    def test_str_renders_small_matrix(self):
        mat = pycauset.causalmatrix(4, populate=False)
        mat[0, 1] = 1
        mat[1, 2] = 1
        mat[2, 3] = 1
        expected = (
            "causalmatrix(shape=(4, 4))\n"
            "[\n"
            " [0 1 0 0]\n"
            " [0 0 1 0]\n"
            " [0 0 0 1]\n"
            " [0 0 0 0]\n"
            "]"
        )
        self.assertEqual(str(mat), expected)

    def test_str_truncates_large_matrix(self):
        mat = pycauset.causalmatrix(12, populate=False)
        mat[0, 11] = 1
        view = str(mat)
        self.assertIn("causalmatrix(shape=(12, 12))", view)
        self.assertIn("...", view)
        # Ensure leading rows are listed
        self.assertIn("[0 0 0 ... 0 0 1]", view)

    def test_matrix_str_uses_matrix_label(self):
        mat = pycauset.matrix(3, populate=False)
        view = str(mat)
        self.assertTrue(view.startswith("matrix(shape=(3, 3))"))

    def test_populate_sets_random_entries(self):
        attempts = 10
        for _ in range(attempts):
            mat = pycauset.causalmatrix(6)
            found = False
            for i in range(6):
                for j in range(i + 1, 6):
                    if mat[i, j]:
                        found = True
                        break
                if found:
                    break
            if found:
                return
        self.fail("populate=True did not produce any edges across attempts")

    def test_matrix_accepts_nested_lists(self):
        data = [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
        mat = pycauset.matrix(data, populate=False)
        self.assertEqual(mat[0, 1], 1)
        self.assertEqual(mat[1, 2], 1)
        self.assertEqual(mat[2, 3], 1)
        self.assertEqual(mat[0, 3], 0)

    def test_matrix_accepts_numpy_arrays(self):
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - NumPy optional
            self.skipTest("NumPy not available")

        arr = np.zeros((4, 4), dtype=bool)
        arr[0, 2] = True
        arr[1, 3] = True
        mat = pycauset.matrix(arr)
        self.assertEqual(mat[0, 2], 1)
        self.assertEqual(mat[1, 3], 1)
        self.assertEqual(mat[2, 3], 0)

    def test_matrix_accepts_lower_triangular_sequences(self):
        data = [
            [1, 0, 0],
            [2, 3, 0],
            [4, 5, 6],
        ]
        mat = pycauset.matrix(data)
        self.assertEqual(mat[2, 0], 4)
        self.assertEqual(mat[2, 2], 6)

    def test_matrix_allows_arbitrary_numeric_values(self):
        mat = pycauset.matrix([
            [0.1, 2.5],
            [-3.2, 7.7],
        ])
        self.assertAlmostEqual(mat[0, 1], 2.5)
        self.assertAlmostEqual(mat[1, 0], -3.2)

    def test_seed_controls_population(self):
        pycauset.seed = 123

        def snapshot(matrix: pycauset.causalmatrix) -> tuple[int, ...]:
            size = matrix.size()
            bits: list[int] = []
            for i in range(size):
                for j in range(i + 1, size):
                    bits.append(matrix[i, j])
            return tuple(bits)

        first = pycauset.causalmatrix(6)
        second = pycauset.causalmatrix(6)
        self.assertEqual(snapshot(first), snapshot(second))

        pycauset.seed = 321
        third = pycauset.causalmatrix(6)
        self.assertNotEqual(snapshot(first), snapshot(third))


if __name__ == "__main__":
    unittest.main()
