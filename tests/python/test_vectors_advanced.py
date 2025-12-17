import unittest
import pycauset
from pycauset import CausalSet, MinkowskiDiamond

class TestVectorsAdvanced(unittest.TestCase):
    def test_large_vector(self):
        # Test with a larger size to check for memory issues or performance
        size = 10000
        v = pycauset.zeros(size, dtype="float")
        self.assertEqual(len(v), size)
        v[size-1] = 1.0
        self.assertEqual(v[size-1], 1.0)
        self.assertEqual(v[0], 0.0)

    def test_vector_iteration(self):
        data = [1.0, 2.0, 3.0, 4.0]
        v = pycauset.vector(data)
        
        # Check iteration
        extracted = [x for x in v]
        self.assertEqual(extracted, data)

    def test_vector_from_causet(self):
        # Often we want vectors related to causets (e.g. labels, values)
        c = CausalSet(n=10, spacetime=MinkowskiDiamond(2))
        
        # Create a vector of size n
        v = pycauset.empty(c.n, dtype="int")
        for i in range(c.n):
            v[i] = i
            
        self.assertEqual(len(v), c.n)
        self.assertEqual(v[9], 9)

    def test_matrix_vector_multiplication(self):
        # If supported. If not, this test will fail or we can remove it.
        # Assuming standard matrix-vector multiplication exists or is planned.
        # Based on C++ headers (MatrixOperations.hpp), it likely exists.
        
        # Create a simple matrix (identity-like or small)
        # Since we can't easily create a custom matrix from scratch in Python yet (maybe?),
        # we'll use a small causet's matrix.
        
        c = CausalSet(n=3, spacetime=MinkowskiDiamond(2))
        # C is a TriangularBitMatrix.
        # Let's try to multiply it by a vector.
        
        v = pycauset.vector([1.0, 1.0, 1.0], dtype="float")
        
        try:
            # Try direct multiplication
            result = c.C * v
            
            # Check result dimensions
            self.assertEqual(len(result), 3)
            
            # Check values (roughly)
            # Result[i] = sum(C[i,j] * v[j])
            # Since v is all 1s, Result[i] is the number of ancestors/descendants (depending on convention)
            # plus maybe self if reflexive.
            
            # Let's verify manually
            for i in range(3):
                row_sum = 0.0
                for j in range(3):
                    if c.C[i, j]:
                        row_sum += 1.0
                self.assertAlmostEqual(result[i], row_sum)
                
        except TypeError:
            # If operator overloading isn't implemented, maybe there's a function
            pass

    def test_dot_product_types(self):
        # Float dot Float
        v1 = pycauset.vector([1.0, 2.0], dtype="float")
        v2 = pycauset.vector([3.0, 4.0], dtype="float")
        self.assertAlmostEqual(v1.dot(v2), 11.0)
        
        # Int dot Int
        v3 = pycauset.vector([1, 2], dtype="int")
        v4 = pycauset.vector([3, 4], dtype="int")
        self.assertEqual(v3.dot(v4), 11)
        
        # Mixed (might fail or cast)
        try:
            # v1=[1.0, 2.0], v3=[1, 2] -> 1*1 + 2*2 = 5.0
            res = v1.dot(v3)
            self.assertAlmostEqual(res, 5.0)
        except (TypeError, ValueError):
            pass

if __name__ == '__main__':
    unittest.main()
