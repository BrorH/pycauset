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
                self.assertEqual(loaded_matrix.rows(), n)
                self.assertEqual(loaded_matrix.cols(), n)
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
                self.assertEqual(loaded_c.C.rows(), 50)
                self.assertEqual(loaded_c.C.cols(), 50)
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
                self.assertEqual(loaded.rows(), n)
                self.assertEqual(loaded.cols(), n)
            finally:
                loaded.close()
        finally:
            matrix.close()

    def test_dense_bit_matrix_rectangular_save_load(self):
        if getattr(pycauset, "DenseBitMatrix", None) is None:
            self.skipTest("DenseBitMatrix is not available")

        m = pycauset.DenseBitMatrix(3, 5)
        m.set(1, 4, True)
        m.set(2, 0, True)
        path = self.test_dir / "bit_matrix_rect.pycauset"

        try:
            pycauset.save(m, path)

            with zipfile.ZipFile(path, "r") as zf:
                with zf.open("metadata.json") as f:
                    meta = json.load(f)
                    self.assertEqual(meta["rows"], 3)
                    self.assertEqual(meta["cols"], 5)
                    self.assertEqual(meta.get("data_type"), "BIT")

            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, pycauset.DenseBitMatrix)
                self.assertEqual(loaded.rows(), 3)
                self.assertEqual(loaded.cols(), 5)
                self.assertTrue(loaded.get(1, 4))
                self.assertTrue(loaded.get(2, 0))
                self.assertFalse(loaded.get(0, 0))
            finally:
                loaded.close()
        finally:
            m.close()

    def test_square_only_metadata_mismatch_rejected(self):
        n = 16
        matrix = TriangularBitMatrix.random(n, 0.25, seed=7)
        path = self.test_dir / "triangular_ok.pycauset"
        bad_path = self.test_dir / "triangular_bad_cols.pycauset"

        try:
            pycauset.save(matrix, path)

            with zipfile.ZipFile(path, "r") as zf:
                meta = json.loads(zf.read("metadata.json").decode("utf-8"))
                data = zf.read("data.bin")

            # Corrupt the shape: triangular/causal matrices are square-only.
            meta["cols"] = int(n + 1)

            with zipfile.ZipFile(bad_path, "w", zipfile.ZIP_STORED) as zf:
                zf.writestr("metadata.json", json.dumps(meta, indent=2))
                zf.writestr("data.bin", data)

            with self.assertRaises(ValueError):
                pycauset.load(bad_path)
        finally:
            matrix.close()

    def test_int16_matrix_save_load(self):
        if getattr(pycauset, "Int16Matrix", None) is None:
            self.skipTest("Int16Matrix is not available")

        n = 10
        matrix = pycauset.Int16Matrix(n)
        matrix[0, 0] = 7
        matrix[1, 0] = -3
        path = self.test_dir / "int16_matrix.pycauset"

        try:
            pycauset.save(matrix, path)

            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, pycauset.Int16Matrix)
                self.assertEqual(loaded.rows(), n)
                self.assertEqual(loaded.cols(), n)
                self.assertEqual(loaded[0, 0], 7)
                self.assertEqual(loaded[1, 0], -3)
            finally:
                loaded.close()
        finally:
            matrix.close()

    def test_int16_vector_save_load(self):
        if getattr(pycauset, "Int16Vector", None) is None:
            self.skipTest("Int16Vector is not available")

        n = 10
        vec = pycauset.Int16Vector(n)
        vec[0] = 4
        vec[1] = -5
        path = self.test_dir / "int16_vector.pycauset"

        try:
            pycauset.save(vec, path)

            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, pycauset.Int16Vector)
                self.assertEqual(len(loaded), n)
                self.assertEqual(loaded[0], 4)
                self.assertEqual(loaded[1], -5)
            finally:
                loaded.close()
        finally:
            vec.close()

if __name__ == "__main__":
    unittest.main()
