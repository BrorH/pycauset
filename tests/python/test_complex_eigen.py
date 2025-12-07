import unittest
import pycauset
import numpy as np

class TestComplexEigen(unittest.TestCase):
    def test_complex_eigvals(self):
        # Create a simple complex matrix
        # [ 1  i ]
        # [ -i 1 ]
        # Eigenvalues should be 0 and 2
        
        # Assuming ComplexMatrix constructor takes rows, cols
        # Or maybe we construct it from real and imag parts?
        # Let's try to find out how to construct it.
        # For now, I'll assume we can create it and set elements.
        
        # Actually, let's look at how ComplexMatrix is exposed.
        # But for now, I'll try to use the factory or constructor.
        
        n = 2
        # If there is no direct constructor from numpy, we might need to set elements
        # But let's try to see if we can use a helper or just set elements.
        
        # Let's assume we can create it like this:
        cm = pycauset.ComplexMatrix(n)
        
        # Set elements
        # (0,0) = 1 + 0j
        cm.set(0, 0, complex(1.0, 0.0))
        # (0,1) = 0 + 1j
        cm.set(0, 1, complex(0.0, 1.0))
        # (1,0) = 0 - 1j
        cm.set(1, 0, complex(0.0, -1.0))
        # (1,1) = 1 + 0j
        cm.set(1, 1, complex(1.0, 0.0))
        
        # Compute eigenvalues
        evals = pycauset.eigvals(cm)
        
        # evals should be a ComplexVector or similar, which might be exposed as a list or object
        # Let's assume it returns an object that we can convert to list or has accessors
        
        # If it returns a ComplexVector, we might need to iterate
        print("Eigenvalues type:", type(evals))
        
        # Let's try to convert to numpy array if possible, or iterate
        # Assuming it has a size() and get(i) or similar
        
        vals = []
        for i in range(evals.size()):
            val = evals.get(i) # returns tuple (real, imag) or complex?
            # If it returns tuple
            if isinstance(val, tuple):
                vals.append(complex(val[0], val[1]))
            else:
                vals.append(val)
                
        vals = np.array(vals)
        vals.sort() # Sort to compare
        
        expected = np.array([0.0, 2.0])
        expected.sort()
        
        print("Computed eigenvalues:", vals)
        print("Expected eigenvalues:", expected)
        
        self.assertTrue(np.allclose(vals, expected, atol=1e-5))

    def test_complex_eig(self):
        # Test full eig decomposition
        n = 2
        cm = pycauset.ComplexMatrix(n)
        
        cm.set(0, 0, complex(1.0, 0.0))
        cm.set(0, 1, complex(0.0, 1.0))
        cm.set(1, 0, complex(0.0, -1.0))
        cm.set(1, 1, complex(1.0, 0.0))
        
        vals, vecs = pycauset.eig(cm)
        
        # Check eigenvalues
        val_list = []
        for i in range(vals.size()):
            val_list.append(vals.get(i))
        
        val_arr = np.array(val_list)
        # Sorting complex numbers is tricky, but here they are real.
        # However, let's just check if they match expected set.
        val_arr_sorted = np.sort(val_arr)
        
        expected_vals = np.array([0.0, 2.0])
        expected_vals.sort()
        
        self.assertTrue(np.allclose(val_arr_sorted, expected_vals, atol=1e-5))
        
        # Check eigenvectors A*v = lambda*v
        for i in range(n):
            # Get i-th eigenvalue (we need to find which one corresponds to column i)
            # Eigen returns vals and vecs such that vecs[:,i] corresponds to vals[i]
            lam = vals.get(i)
            
            # Get i-th eigenvector (column i of vecs)
            v = np.zeros(n, dtype=complex)
            for r in range(n):
                v[r] = vecs.get(r, i)
                
            # Compute A*v
            Av = np.zeros(n, dtype=complex)
            for r in range(n):
                row_sum = 0j
                for c in range(n):
                    row_sum += cm.get(r, c) * v[c]
                Av[r] = row_sum
                
            # Compute lambda*v
            lam_v = lam * v
            
            # Check Av == lam_v
            self.assertTrue(np.allclose(Av, lam_v, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
