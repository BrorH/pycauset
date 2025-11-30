import pycauset
import unittest
import os

class TestTransposedVectors(unittest.TestCase):
    def test_transpose_properties(self):
        v = pycauset.FloatVector(5)
        for i in range(5):
            v[i] = i
            
        # Check initial shape
        self.assertEqual(v.shape, (5,))
        
        # Transpose
        vt = v.T
        self.assertEqual(vt.shape, (1, 5))
        self.assertTrue("transposed=True" in str(vt))
        
        # Double Transpose
        vtt = vt.T
        self.assertEqual(vtt.shape, (5,))
        self.assertFalse("transposed=True" in str(vtt))
        
        # Check values
        self.assertEqual(vt[0], 0.0)
        self.assertEqual(vt[4], 4.0)

    def test_outer_product(self):
        v1 = pycauset.FloatVector(3)
        v1[0] = 1; v1[1] = 2; v1[2] = 3
        
        v2 = pycauset.FloatVector(3)
        v2[0] = 4; v2[1] = 5; v2[2] = 6
        
        # v1 @ v2.T -> Outer Product
        # v1 is (3,), v2.T is (1, 3).
        # Numpy: (3,) @ (1, 3) -> Error?
        # Numpy: outer(a, b) is a[:,None] @ b[None,:] -> (3,1) @ (1,3) -> (3,3)
        # Our implementation: Col @ Row -> Outer.
        # v1 is Col (implicitly), v2.T is Row.
        
        m = v1 @ v2.T
        
        # Check result type
        # Should be FloatMatrix
        # self.assertIsInstance(m, pycauset.FloatMatrix) # Need to check exact type name
        
        self.assertEqual(m.shape, (3, 3))
        self.assertEqual(m[0, 0], 4.0) # 1*4
        self.assertEqual(m[0, 1], 5.0) # 1*5
        self.assertEqual(m[1, 0], 8.0) # 2*4
        self.assertEqual(m[2, 2], 18.0) # 3*6

    def test_inner_product(self):
        v1 = pycauset.FloatVector(3)
        v1[0] = 1; v1[1] = 2; v1[2] = 3
        
        # v1.T @ v1 -> Scalar
        s = v1.T @ v1
        self.assertEqual(s, 14.0) # 1+4+9
        
        # v1 @ v1 -> Scalar (Numpy behavior for 1D)
        s2 = v1 @ v1
        self.assertEqual(s2, 14.0)

    def test_matrix_vector_mul(self):
        m = pycauset.FloatMatrix(2)
        m[0,0] = 1; m[0,1] = 2
        m[1,0] = 3; m[1,1] = 4
        
        v = pycauset.FloatVector(2)
        v[0] = 1; v[1] = 2
        
        # M @ v -> Col Vector
        res = m @ v
        self.assertEqual(res.shape, (2,))
        self.assertEqual(res[0], 5.0) # 1*1 + 2*2
        self.assertEqual(res[1], 11.0) # 3*1 + 4*2
        
        # v.T @ M -> Row Vector
        res2 = v.T @ m
        self.assertEqual(res2.shape, (1, 2))
        self.assertEqual(res2[0], 7.0) # 1*1 + 2*3
        self.assertEqual(res2[1], 10.0) # 1*2 + 2*4

    def test_persistence(self):
        v = pycauset.FloatVector(3)
        v[0] = 10
        vt = v.T
        
        # Save transposed vector
        path = "test_transposed.pycauset"
        if os.path.exists(path):
            os.remove(path)
            
        pycauset.save(vt, path)
        
        # Load it back
        vt_loaded = pycauset.load(path)
        self.assertEqual(vt_loaded.shape, (1, 3))
        self.assertEqual(vt_loaded[0], 10.0)
        
        # Cleanup
        vt_loaded.close()
        if os.path.exists(path):
            os.remove(path)

if __name__ == '__main__':
    unittest.main()
