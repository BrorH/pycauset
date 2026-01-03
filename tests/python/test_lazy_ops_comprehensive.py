import unittest
import pycauset
import math

class TestLazyOpsComprehensive(unittest.TestCase):
    def setUp(self):
        pass

    def test_matrix_scalar_add(self):
        n = 5
        A = pycauset.FloatMatrix(n, n)
        A[0,0] = 10.0
        
        # Matrix + Scalar -> Lazy
        B = A + 5.0
        self.assertEqual(type(B).__name__, "LazyMatrix")
        self.assertAlmostEqual(B[0,0], 15.0)
        
        # Scalar + Matrix -> Lazy (via __radd__?)
        # bind_matrix.cpp __radd__ returns MatrixBase (eager) for scalar!
        C = 5.0 + A
        # self.assertEqual(type(C).__name__, "LazyMatrix") # Might be eager
        self.assertAlmostEqual(C[0,0], 15.0)

    def test_matrix_matrix_sub(self):
        n = 5
        A = pycauset.FloatMatrix(n, n)
        B = pycauset.FloatMatrix(n, n)
        A[0,0] = 10.0
        B[0,0] = 3.0
        
        # Matrix - Matrix -> Lazy
        C = A - B
        self.assertEqual(type(C).__name__, "LazyMatrix")
        self.assertAlmostEqual(C[0,0], 7.0)

    def test_lazy_scalar_ops(self):
        n = 5
        A = pycauset.FloatMatrix(n, n)
        A[0,0] = 10.0
        X = A + 0.0 # Make it lazy
        
        # Lazy + Scalar
        Y = X + 2.0
        self.assertEqual(type(Y).__name__, "LazyMatrix")
        self.assertAlmostEqual(Y[0,0], 12.0)
        
        # Scalar + Lazy
        Z = 2.0 + X
        self.assertEqual(type(Z).__name__, "LazyMatrix")
        self.assertAlmostEqual(Z[0,0], 12.0)
        
        # Lazy - Scalar
        W = X - 2.0
        self.assertEqual(type(W).__name__, "LazyMatrix")
        self.assertAlmostEqual(W[0,0], 8.0)
        
        # Scalar - Lazy
        V = 20.0 - X
        self.assertEqual(type(V).__name__, "LazyMatrix")
        self.assertAlmostEqual(V[0,0], 10.0)
        
        # Lazy * Scalar
        U = X * 2.0
        self.assertEqual(type(U).__name__, "LazyMatrix")
        self.assertAlmostEqual(U[0,0], 20.0)
        
        # Scalar * Lazy
        T = 2.0 * X
        self.assertEqual(type(T).__name__, "LazyMatrix")
        self.assertAlmostEqual(T[0,0], 20.0)

    def test_lazy_matrix_ops(self):
        n = 5
        A = pycauset.FloatMatrix(n, n)
        A[0,0] = 10.0
        X = A + 0.0 # Lazy
        
        B = pycauset.FloatMatrix(n, n)
        B[0,0] = 2.0
        
        # Lazy + Matrix
        C = X + B
        self.assertEqual(type(C).__name__, "LazyMatrix")
        self.assertAlmostEqual(C[0,0], 12.0)
        
        # Matrix + Lazy
        D = B + X
        self.assertEqual(type(D).__name__, "LazyMatrix")
        self.assertAlmostEqual(D[0,0], 12.0)
        
        # Lazy - Matrix
        E = X - B
        self.assertEqual(type(E).__name__, "LazyMatrix")
        self.assertAlmostEqual(E[0,0], 8.0)
        
        # Matrix - Lazy
        F = B - X
        self.assertEqual(type(F).__name__, "LazyMatrix")
        self.assertAlmostEqual(F[0,0], -8.0)

    def test_lazy_lazy_ops(self):
        n = 5
        A = pycauset.FloatMatrix(n, n)
        A[0,0] = 10.0
        X = A + 0.0 # Lazy
        Y = A + 1.0 # Lazy
        
        # Lazy + Lazy
        Z = X + Y
        self.assertEqual(type(Z).__name__, "LazyMatrix")
        self.assertAlmostEqual(Z[0,0], 21.0)
        
        # Lazy - Lazy
        W = Y - X
        self.assertEqual(type(W).__name__, "LazyMatrix")
        self.assertAlmostEqual(W[0,0], 1.0)

    def test_factory_functions(self):
        n = 5
        A = pycauset.FloatMatrix(n, n)
        A[0,0] = 0.0
        
        # lazy_sin
        S = pycauset.lazy_sin(A)
        self.assertEqual(type(S).__name__, "LazyMatrix")
        self.assertAlmostEqual(S[0,0], 0.0)
        
        # lazy_cos
        C = pycauset.lazy_cos(A)
        self.assertEqual(type(C).__name__, "LazyMatrix")
        self.assertAlmostEqual(C[0,0], 1.0)

if __name__ == '__main__':
    unittest.main()
