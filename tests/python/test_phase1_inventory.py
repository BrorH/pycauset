import unittest
import numpy as np
import pycauset as pc
import os
from pathlib import Path

class TestPhase1Inventory(unittest.TestCase):
    def test_01_lazy_arrays_materialize(self):
        """Verify that np.asarray(Expression) triggers evaluation."""
            
        A = pc.zeros((2, 2), dtype=pc.float64)
        B = pc.zeros((2, 2), dtype=pc.float64)
        A[0,0] = 1.0
        B[0,0] = 2.0
        
        # This returns an Expression (BinaryExpression)
        C_expr = A + B
        
        # DEBUG: Check manual invocation
        try:
            res = C_expr.__array__()
            print(f"DEBUG: __array__ returned type: {type(res)}")
            print(f"DEBUG: __array__ returned value: {res}")
            # Verify result is convertible
            res_arr = np.array(res)
            print(f"DEBUG: np.array(res) shape: {res_arr.shape}")
        except Exception as e:
            print(f"DEBUG: Manual __array__ call failed: {e}")
        C_arr = np.array(C_expr)
        
        self.assertEqual(C_arr.shape, (2, 2))
        self.assertEqual(C_arr[0, 0], 3.0)
        
    def test_02_snapshot_export_loads_data(self):
        """Reproduce failure: Verify snapshot-based export reads data."""
        m = pc.DenseMatrix(10, 10, dtype="float64")
        m[0, 0] = 42.0
        snap_path = "inventory_test.pycauset"
        try:
            pc.save(m, snap_path)
            loaded = pc.load(snap_path)
            
            # This failed in test_numpy_conversion_policy
            arr = np.array(loaded)
            
            self.assertEqual(arr.shape, (10, 10))
            self.assertEqual(arr[0, 0], 42.0)
        finally:
             if os.path.exists(snap_path):
                 try: os.remove(snap_path) 
                 except: pass

    def test_03_every_dtype_exports_with_its_numpy_dtype(self):
        """Every dense dtype has __array__ and exports with the matching NumPy dtype."""
        pairs = {
            "bool": np.bool_, "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
            "uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32, "uint64": np.uint64,
            "float16": np.float16, "float32": np.float32, "float64": np.float64,
            "complex_float32": np.complex64, "complex_float64": np.complex128,
        }
        for dt, np_dt in pairs.items():
            m = pc.DenseMatrix(2, 2, dtype=dt)
            self.assertTrue(hasattr(m, "__array__"))
            self.assertEqual(np.array(m).dtype, np_dt, dt)
            v = pc.DenseVector(2, dtype=dt)
            self.assertEqual(np.array(v).dtype, np_dt, dt)

if __name__ == '__main__':
    unittest.main()
