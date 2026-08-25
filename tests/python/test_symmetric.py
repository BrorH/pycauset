import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

_STORAGE_TMP = tempfile.TemporaryDirectory()
import pycauset as pc

pc.set_backing_dir(_STORAGE_TMP.name)


def _sym(n=3):
    return np.array(
        [[2.0, 1.0, 3.0], [1.0, 5.0, 0.0], [3.0, 0.0, 7.0]]
    )


def _anti(n=3):
    return np.array(
        [[0.0, 2.0, -1.0], [-2.0, 0.0, 4.0], [1.0, -4.0, 0.0]]
    )


class TestSymmetricFactory(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        _STORAGE_TMP.cleanup()

    def test_symmetric_float64_native_type(self):
        A = _sym()
        m = pc.symmetric(A)
        self.assertIsInstance(m, pc.SymmetricMatrix)
        self.assertEqual(m.shape, (3, 3))
        self.assertTrue(m.properties["is_symmetric"])

    def test_symmetric_roundtrip_export(self):
        A = _sym()
        m = pc.symmetric(A)
        self.assertTrue(np.array_equal(pc.to_numpy(m), A))
        self.assertTrue(np.array_equal(np.asarray(m), A))

    def test_antisymmetric_float64_native_type(self):
        B = _anti()
        m = pc.antisymmetric(B)
        self.assertIsInstance(m, pc.AntiSymmetricMatrix)
        self.assertEqual(m.shape, (3, 3))
        self.assertTrue(m.properties["is_anti_symmetric"])
        self.assertTrue(m.properties["has_zero_diagonal"])

    def test_antisymmetric_roundtrip_export(self):
        B = _anti()
        m = pc.antisymmetric(B)
        self.assertTrue(np.array_equal(pc.to_numpy(m), B))

    def test_symmetric_rejects_non_symmetric(self):
        with self.assertRaises(ValueError):
            pc.symmetric(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_antisymmetric_rejects_symmetric(self):
        with self.assertRaises(ValueError):
            pc.antisymmetric(np.array([[1.0, 2.0], [2.0, 1.0]]))

    def test_antisymmetric_rejects_nonzero_diagonal(self):
        with self.assertRaises(ValueError):
            pc.antisymmetric(np.array([[1.0, 2.0], [-2.0, 0.0]]))

    def test_symmetric_rejects_non_square(self):
        with self.assertRaises(ValueError):
            pc.symmetric(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))

    def test_symmetric_rejects_complex(self):
        with self.assertRaises(TypeError):
            pc.symmetric(np.array([[1 + 0j, 0j], [0j, 1 + 0j]]))

    def test_symmetric_float32_promotes_to_float64(self):
        A32 = _sym().astype(np.float32)
        m = pc.symmetric(A32)
        self.assertIsInstance(m, pc.SymmetricMatrix)
        self.assertTrue(np.allclose(pc.to_numpy(m), A32.astype(np.float64)))

    def test_symmetric_integer_dense_with_property(self):
        A = np.array([[1, 2], [2, 3]], dtype=np.int32)
        m = pc.symmetric(A)
        self.assertIsInstance(m, pc.IntegerMatrix)
        self.assertTrue(m.properties["is_symmetric"])
        self.assertTrue(np.array_equal(pc.to_numpy(m), A))

    def test_antisymmetric_integer_dense_with_property(self):
        B = np.array([[0, 2], [-2, 0]], dtype=np.int32)
        m = pc.antisymmetric(B)
        self.assertTrue(m.properties["is_anti_symmetric"])
        self.assertTrue(m.properties["has_zero_diagonal"])
        self.assertTrue(np.array_equal(pc.to_numpy(m), B))

    def test_symmetric_bool_dense(self):
        A = np.array([[True, False], [False, True]], dtype=np.bool_)
        m = pc.symmetric(A)
        self.assertTrue(m.properties["is_symmetric"])
        self.assertTrue(np.array_equal(pc.to_numpy(m), A))

    def test_symmetric_and_antisymmetric_in_all(self):
        self.assertIn("symmetric", pc.__all__)
        self.assertIn("antisymmetric", pc.__all__)

    def test_packed_storage_is_half_of_dense(self):
        n = 2000
        s = pc.SymmetricMatrix(n)
        d = pc.FloatMatrix(n)
        try:
            with tempfile.TemporaryDirectory() as td:
                sp = Path(td) / "s.raw"
                dp = Path(td) / "d.raw"
                s.copy_storage(str(sp))
                d.copy_storage(str(dp))
                ratio = sp.stat().st_size / dp.stat().st_size
                # Packed upper triangle is n(n+1)/2 vs n^2, i.e. ~2x smaller
                # (the 64-byte header is negligible at this size).
                self.assertAlmostEqual(ratio, 0.5, delta=0.02)
        finally:
            s.close()
            d.close()


class TestSymmetricOps(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        _STORAGE_TMP.cleanup()

    def test_matmul_sym_sym(self):
        A = _sym()
        m = pc.symmetric(A)
        r = pc.matmul(m, m)
        self.assertTrue(np.allclose(pc.to_numpy(r), A @ A))

    def test_matmul_sym_dense(self):
        A = _sym()
        D = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
        m = pc.symmetric(A)
        r = pc.matmul(m, pc.matrix(D))
        self.assertTrue(np.allclose(pc.to_numpy(r), A @ D))

    def test_matmul_identity_sym(self):
        A = _sym()
        m = pc.symmetric(A)
        I3 = pc.identity(3)
        self.assertTrue(np.allclose(pc.to_numpy(pc.matmul(I3, m)), A))
        self.assertTrue(np.allclose(pc.to_numpy(pc.matmul(m, I3)), A))

    def test_add_sym_sym(self):
        A = _sym()
        m = pc.symmetric(A)
        r = m + m
        self.assertTrue(np.allclose(pc.to_numpy(r), A + A))

    def test_trace_sym(self):
        A = _sym()
        m = pc.symmetric(A)
        self.assertAlmostEqual(pc.trace(m), float(np.trace(A)))

    def test_trace_antisymmetric_zero(self):
        m = pc.antisymmetric(_anti())
        self.assertAlmostEqual(pc.trace(m), 0.0)

    def test_determinant_sym(self):
        A = _sym()
        m = pc.symmetric(A)
        self.assertAlmostEqual(pc.determinant(m), float(np.linalg.det(A)))

    def test_determinant_antisymmetric_odd_zero(self):
        m = pc.antisymmetric(_anti())
        self.assertAlmostEqual(pc.determinant(m), 0.0)

    def test_eigvalsh_sym(self):
        A = _sym()
        m = pc.symmetric(A)
        got = np.asarray(pc.eigvalsh(m))
        want = np.linalg.eigvalsh(A)
        self.assertTrue(np.allclose(np.sort(got), np.sort(want)))

    def test_norm_sym(self):
        A = _sym()
        m = pc.symmetric(A)
        self.assertAlmostEqual(pc.norm(m), float(np.linalg.norm(A)))

    def test_matrix_rank_sym(self):
        A = _sym()
        m = pc.symmetric(A)
        self.assertEqual(pc.matrix_rank(m), int(np.linalg.matrix_rank(A)))

    def test_transpose_sym_is_self(self):
        A = _sym()
        m = pc.symmetric(A)
        self.assertTrue(np.allclose(pc.to_numpy(m.T), A))

    def test_transpose_antisymmetric_is_negation(self):
        B = _anti()
        m = pc.antisymmetric(B)
        self.assertTrue(np.allclose(pc.to_numpy(m.T), -B))

    def test_scalar_multiply(self):
        A = _sym()
        m = pc.symmetric(A)
        self.assertTrue(np.allclose(pc.to_numpy(2.0 * m), 2.0 * A))


class TestSymmetricPersistence(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        _STORAGE_TMP.cleanup()

    def test_symmetric_roundtrip(self):
        A = _sym()
        m = pc.symmetric(A)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sym.pycauset"
            pc.save(m, path)
            loaded = pc.load(path)
            try:
                self.assertIsInstance(loaded, pc.SymmetricMatrix)
                self.assertTrue(np.array_equal(pc.to_numpy(loaded), A))
            finally:
                loaded.close()

    def test_antisymmetric_roundtrip(self):
        B = _anti()
        m = pc.antisymmetric(B)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "anti.pycauset"
            pc.save(m, path)
            loaded = pc.load(path)
            try:
                self.assertIsInstance(loaded, pc.AntiSymmetricMatrix)
                self.assertTrue(np.array_equal(pc.to_numpy(loaded), B))
            finally:
                loaded.close()


class TestDiagonalFactory(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        _STORAGE_TMP.cleanup()

    def test_diagonal_from_vector_float(self):
        d = pc.diagonal([1.0, 2.0, 3.0])
        self.assertIsInstance(d, pc.DiagonalMatrix)
        self.assertTrue(d.properties["is_diagonal"])
        self.assertTrue(np.array_equal(pc.to_numpy(d), np.diag([1.0, 2.0, 3.0])))

    def test_diagonal_from_matrix_float(self):
        d = pc.diagonal(np.array([[1.0, 9.0], [9.0, 2.0]]))
        self.assertIsInstance(d, pc.DiagonalMatrix)
        self.assertTrue(np.array_equal(pc.to_numpy(d), np.diag([1.0, 2.0])))

    def test_diagonal_from_vector_int_dense(self):
        d = pc.diagonal(np.array([1, 2, 3], dtype=np.int32))
        self.assertTrue(d.properties["is_diagonal"])
        self.assertTrue(np.array_equal(pc.to_numpy(d), np.diag([1, 2, 3])))

    def test_diagonal_rejects_non_square(self):
        with self.assertRaises(ValueError):
            pc.diagonal(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_diagonal_structural_shortcuts(self):
        d = pc.diagonal([1.0, 2.0, 3.0])
        self.assertAlmostEqual(pc.trace(d), 6.0)
        self.assertAlmostEqual(pc.determinant(d), 6.0)
        self.assertEqual(pc.matrix_rank(d), 3)

    def test_diagonal_persistence_roundtrip(self):
        d = pc.diagonal([1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "diag.pycauset"
            pc.save(d, path)
            loaded = pc.load(path)
            try:
                self.assertIsInstance(loaded, pc.DiagonalMatrix)
                self.assertTrue(np.array_equal(pc.to_numpy(loaded), np.diag([1.0, 2.0, 3.0])))
            finally:
                loaded.close()


if __name__ == "__main__":
    unittest.main()
