
import pytest
import numpy as np
import pycauset as la
from pycauset import matrix

def test_cholesky():
    # Construct SPD matrix
    np_a = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float64)
    a = matrix(np_a)
    
    l = la.cholesky(a)
    np_l = np.linalg.cholesky(np_a)
    
    # Check L * L.T close to A
    diff = np.linalg.norm(np.array(l) - np_l)
    assert diff < 1e-10

def test_qr():
    np_a = np.random.rand(10, 5).astype(np.float64)
    a = matrix(np_a)
    
    q, r = la.qr(a)
    
    # Check dimensions
    assert q.rows() == 10
    assert q.cols() == 5
    assert r.rows() == 5
    assert r.cols() == 5
    
    # Check orthogonality of Q: Q.T @ Q = I
    q_np = np.array(q)
    qtq = np.dot(q_np.T, q_np)
    assert np.allclose(qtq, np.eye(5), atol=1e-10)
    
    # Check reconstruction: A = Q @ R
    r_np = np.array(r)
    recon = np.dot(q_np, r_np)
    assert np.allclose(recon, np_a, atol=1e-10)

def test_solve():
    np_a = np.random.rand(5, 5).astype(np.float64)
    # Ensure non-singular
    np_a += np.eye(5) * 5 
    np_b = np.random.rand(5, 2).astype(np.float64)
    
    a = matrix(np_a)
    b = matrix(np_b)
    
    x = la.solve(a, b)
    
    # Check A * X = B
    recon = np.dot(np_a, np.array(x))
    assert np.allclose(recon, np_b, atol=1e-10)

def test_svd():
    np_a = np.random.rand(10, 5).astype(np.float64)
    a = matrix(np_a)
    
    # Request reduced SVD to test native path if available (ops.py routes based on this)
    # And check reduced dimensions
    u, s, vt = la.svd(a, full_matrices=False)
    
    # Dimensions (Reduced SVD)
    assert u.rows() == 10
    assert u.cols() == 5
    assert s.size() == 5
    assert vt.rows() == 5
    assert vt.cols() == 5
    
    # Reconstruct
    S_mat = np.zeros((5, 5))
    np.fill_diagonal(S_mat, np.array(s))
    
    recon = np.dot(np.array(u), np.dot(S_mat, np.array(vt)))
    assert np.allclose(recon, np_a, atol=1e-10)

def test_lu():
    np_a = np.random.rand(5, 5).astype(np.float64)
    a = matrix(np_a)
    
    try:
        p, l_mat, u_mat = la.lu(a)
    except NotImplementedError:
        pytest.skip("LU not implemented/bound")
        return
        
    # Reconstruct A = P @ L @ U
    recon = np.dot(np.array(p), np.dot(np.array(l_mat), np.array(u_mat)))
    assert np.allclose(recon, np_a, atol=1e-10)
    
    # Check L is lower triangular unit
    l_np = np.array(l_mat)
    assert np.allclose(np.tril(l_np), l_np)
    assert np.allclose(np.diag(l_np), 1.0)
    
    # Check U is upper
    u_np = np.array(u_mat)
    assert np.allclose(np.triu(u_np), u_np)
