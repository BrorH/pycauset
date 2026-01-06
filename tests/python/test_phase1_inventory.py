import unittest
import numpy as np
import pycauset as pc
import os
from pathlib import Path

class TestPhase1Inventory(unittest.TestCase):
    def test_01_lazy_arrays_materialize(self):
        """Verify that np.asarray(Expression) triggers evaluation."""
        if not hasattr(pc, "FloatMatrix"):
            self.skipTest("FloatMatrix undefined")
            
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
        m = pc.FloatMatrix(10, 10)
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

    def test_03_enum_python_types(self):
        """List all exportable types."""
        types = [
            pc.FloatVector, pc.IntegerVector,
            pc.FloatMatrix, pc.IntegerMatrix,
            pc.DenseBitMatrix
        ]
        # Check complex if available (R1_COMPLEX not fully active maybe?)
        if hasattr(pc, "ComplexMatrix"):
             types.append(pc.ComplexMatrix)
             
        for T in types:
            inst = T(2) if "Vector" in T.__name__ else T(2, 2)
            has_array = hasattr(inst, "__array__")
            print(f"Type {T.__name__}: __array__={has_array}")
            self.assertTrue(has_array, f"{T.__name__} missing __array__")

    def test_04_dtype_mapping(self):
        """Verify explicit dtype mapping in export."""
        pairs = [
            (pc.FloatMatrix, np.float64),
            (pc.Float32Matrix, np.float32),
            (pc.IntegerMatrix, np.int32),
            (pc.DenseBitMatrix, np.bool), # Should be bool or uint8?
        ]
        for T, np_dt in pairs:
            if not hasattr(pc, T.__name__): continue
            m = T(2, 2)
            arr = np.array(m)
            # Use strict type check or equivalence
            self.assertTrue(arr.dtype == np_dt or np.issubdtype(arr.dtype, np_dt), 
                            f"{T.__name__} -> {arr.dtype}, expected {np_dt}")

if __name__ == '__main__':
    unittest.main()
