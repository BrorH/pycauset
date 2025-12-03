import unittest
import os
import shutil
import zipfile
import json
import struct
from pathlib import Path
import pycauset
import pycauset
from pycauset import CausalSet, TriangularBitMatrix, IntegerMatrix, FloatMatrix

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_storage_tmp")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_matrix_save_load(self):
        # Create a matrix
        n = 100
        matrix = TriangularBitMatrix.random(n, 0.5, seed=42)
        path = self.test_dir / "matrix.pycauset"
        
        try:
            # Save
            pycauset.save(matrix, path)
            
            # Verify file exists
            self.assertTrue(path.exists())
            
            # Verify ZIP structure
            with zipfile.ZipFile(path, "r") as zf:
                self.assertIn("metadata.json", zf.namelist())
                self.assertIn("data.bin", zf.namelist())
                
                # Check metadata
                with zf.open("metadata.json") as f:
                    meta = json.load(f)
                    self.assertEqual(meta["rows"], n)
                    self.assertEqual(meta["matrix_type"], "CAUSAL")
                    
            # Load
            loaded_matrix = pycauset.load(path)
            try:
                self.assertIsInstance(loaded_matrix, TriangularBitMatrix)
                self.assertEqual(loaded_matrix.size(), n)
            finally:
                loaded_matrix.close()
        finally:
            matrix.close()
        
    def test_causet_save_load(self):
        # Create CausalSet
        c = CausalSet(n=50, seed=123)
        path = self.test_dir / "causet.pycauset"
        
        try:
            # Save
            c.save(path)
            
            # Verify ZIP structure
            with zipfile.ZipFile(path, "r") as zf:
                self.assertIn("metadata.json", zf.namelist())
                self.assertIn("data.bin", zf.namelist())
                
                with zf.open("metadata.json") as f:
                    meta = json.load(f)
                    self.assertEqual(meta["object_type"], "CausalSet")
                    self.assertEqual(meta["n"], 50)
                    
            # Load
            loaded_c = CausalSet.load(path)
            try:
                self.assertIsInstance(loaded_c, CausalSet)
                self.assertEqual(loaded_c.n, 50)
                self.assertEqual(loaded_c.C.size(), 50)
            finally:
                loaded_c.C.close()
        finally:
            c.C.close()

    def test_integer_matrix_save_load(self):
        n = 10
        matrix = IntegerMatrix(n) # Zero initialized
        path = self.test_dir / "int_matrix.pycauset"
        
        try:
            pycauset.save(matrix, path)
            
            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, IntegerMatrix)
                self.assertEqual(loaded.size(), n)
            finally:
                loaded.close()
        finally:
            matrix.close()

if __name__ == "__main__":
    unittest.main()
