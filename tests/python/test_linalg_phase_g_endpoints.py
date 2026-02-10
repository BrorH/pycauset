import unittest
import os
import tempfile
import sys
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


class TestPhaseGLinalgEndpoints(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        try:
            pc._runtime.release_tracked_matrices()
        except Exception:
            pass
        try:
            pc.cleanup_storage(Path(_STORAGE_TMP.name))
        except Exception:
            pass
        _STORAGE_TMP.cleanup()

    def test_solve_vector(self):
        A = pc.matrix(((4.0, 1.0), (2.0, 3.0)))
        b = pc.vector((1.0, 0.0))

        x = pc.solve(A, b)

        x_np = np.asarray(x)
        expected = np.linalg.solve(np.asarray(A), np.asarray(b))
        np.testing.assert_allclose(x_np, expected, rtol=1e-10, atol=1e-12)

    def test_lstsq_overdetermined_vector(self):
        A = pc.matrix(((1.0, 1.0), (1.0, 2.0), (1.0, 3.0)))  # 3x2
        b = pc.vector((1.0, 2.0, 2.0))

        x = pc.lstsq(A, b)

        x_np = np.asarray(x)
        expected, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
        np.testing.assert_allclose(x_np, expected, rtol=1e-8, atol=1e-10)

    def test_slogdet_and_cond(self):
        A = pc.matrix(((4.0, 7.0), (2.0, 6.0)))

        sign, logabs = pc.slogdet(A)
        det = np.linalg.det(np.asarray(A))
        self.assertEqual(sign, 1.0)
        self.assertAlmostEqual(logabs, float(np.log(abs(det))), places=10)

        c = pc.cond(A)
        expected = float(np.linalg.norm(np.asarray(A), ord="fro") * np.linalg.norm(np.linalg.inv(np.asarray(A)), ord="fro"))
        self.assertAlmostEqual(c, expected, places=8)

    def test_eigh_and_eigvalsh(self):
        A = pc.matrix(((2.0, 1.0), (1.0, 2.0)))

        w = pc.eigvalsh(A)
        w2, v = pc.eigh(A)

        w_np = np.asarray(w)
        w2_np = np.asarray(w2)
        v_np = np.asarray(v)

        w_ref = np.linalg.eigvalsh(np.asarray(A))
        w2_ref, v_ref = np.linalg.eigh(np.asarray(A))

        np.testing.assert_allclose(w_np, w_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(w2_np, w2_ref, rtol=1e-10, atol=1e-12)

        # Eigenvectors are unique up to sign; check reconstruction instead.
        A_rec = v_np @ np.diag(w2_np) @ v_np.T
        np.testing.assert_allclose(A_rec, np.asarray(A), rtol=1e-8, atol=1e-10)

    def test_cholesky_factorization(self):
        A = pc.matrix(((4.0, 1.0), (1.0, 3.0)))

        L = pc.cholesky(A)

        L_np = np.asarray(L)
        np.testing.assert_allclose(L_np @ L_np.T, np.asarray(A), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(L_np, np.tril(L_np), rtol=1e-8, atol=1e-10)

    def test_unimplemented_factorizations_raise(self):
        A = pc.matrix(((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(ValueError):
            pc.solve_triangular(A, pc.vector((1.0, 2.0)))
        with self.assertRaises(NotImplementedError):
            pc.lu(A)
        with self.assertRaises(NotImplementedError):
            pc.svd(A)
        with self.assertRaises(NotImplementedError):
            pc.pinv(A)

    def test_eigvals_arnoldi_real(self):
        A = pc.matrix(((2.0, 1.0), (1.0, 2.0)))

        vals = pc.eigvals_arnoldi(A, k=2, m=2, tol=1e-10)

        vals_np = np.asarray(vals)
        expected = np.linalg.eigvals(np.asarray(A))
        expected_sorted = np.array(sorted(expected, key=lambda x: abs(x), reverse=True))
        np.testing.assert_allclose(vals_np, expected_sorted, rtol=1e-6, atol=1e-8)

    def test_removed_eig_apis_are_deterministic(self):
        A = pc.matrix(((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(NotImplementedError):
            pc.eig(A)
        with self.assertRaises(NotImplementedError):
            pc.eigvals(A)
        with self.assertRaises(NotImplementedError):
            pc.eigvals_skew(A)
