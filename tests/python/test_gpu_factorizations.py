"""GPU factorization tests (CUDA backend).

These force the GPU backend with `force_backend("gpu")` and verify the factorized
results match the reconstruction identities that the CPU path also satisfies. They
are skipped automatically when no CUDA device is present.
"""
import numpy as np
import pytest

import pycauset


def _cuda_available():
    try:
        return bool(pycauset.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_gpu_lu_reconstructs_float64():
    rng = np.random.default_rng(42)
    n = 64
    a_np = rng.standard_normal((n, n)) + n * np.eye(n)  # non-singular
    a = pycauset.matrix(a_np)

    pycauset.cuda.force_backend("gpu")
    try:
        p, l_mat, u = pycauset.lu(a)
    finally:
        pycauset.cuda.force_backend("auto")

    p_np = np.asarray(p)
    l_np = np.asarray(l_mat)
    u_np = np.asarray(u)

    np.testing.assert_allclose(p_np @ l_np @ u_np, a_np, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.tril(l_np), l_np, atol=1e-8)  # unit lower
    np.testing.assert_allclose(np.diag(l_np), 1.0, atol=1e-8)
    np.testing.assert_allclose(np.triu(u_np), u_np, atol=1e-8)  # upper


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_gpu_lu_reconstructs_float32():
    rng = np.random.default_rng(7)
    n = 48
    a_np = (rng.standard_normal((n, n)) + n * np.eye(n)).astype(np.float32)
    a = pycauset.matrix(a_np)

    pycauset.cuda.force_backend("gpu")
    try:
        p, l_mat, u = pycauset.lu(a)
    finally:
        pycauset.cuda.force_backend("auto")

    p_np = np.asarray(p)
    l_np = np.asarray(l_mat)
    u_np = np.asarray(u)

    np.testing.assert_allclose(p_np @ l_np @ u_np, a_np, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.tril(l_np), l_np, atol=1e-4)  # unit lower
    np.testing.assert_allclose(np.diag(l_np), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.triu(u_np), u_np, atol=1e-4)  # upper


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_gpu_lu_rectangular_falls_back_to_cpu():
    # Rectangular input is routed to the CPU path (GPU lu is square-only); the
    # result must still satisfy A = P L U.
    rng = np.random.default_rng(3)
    m, n = 32, 20
    a_np = rng.standard_normal((m, n))
    a = pycauset.matrix(a_np)

    pycauset.cuda.force_backend("gpu")
    try:
        p, l_mat, u = pycauset.lu(a)
    finally:
        pycauset.cuda.force_backend("auto")

    p_np = np.asarray(p)
    l_np = np.asarray(l_mat)
    u_np = np.asarray(u)

    # P is m x m, L is m x k, U is k x n with k = min(m, n).
    k = min(m, n)
    assert p_np.shape == (m, m)
    assert l_np.shape == (m, k)
    assert u_np.shape == (k, n)
    np.testing.assert_allclose(p_np @ l_np @ u_np, a_np, rtol=1e-5, atol=1e-8)
