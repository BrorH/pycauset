import unittest
import os
import shutil
import zipfile
import json
import struct
from pathlib import Path
import pycauset
from pycauset import CausalSet, TriangularBitMatrix, IntegerMatrix, FloatMatrix, MinkowskiCylinder

class TestStorageExtensive(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_storage_extensive_tmp")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        # Force garbage collection to ensure destructors run and close handles
        import gc
        gc.collect()
        
        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                print(f"WARNING: Could not clean up {self.test_dir} due to file locking.")

    def test_large_matrix(self):
        """Test saving and loading a moderately large matrix."""
        n = 1000
        print(f"Generating {n}x{n} matrix...")
        matrix = TriangularBitMatrix.random(n, 0.1, seed=12345)
        path = self.test_dir / "large.pycauset"
        
        try:
            pycauset.save(matrix, path)
            
            loaded = pycauset.load(path)
            try:
                self.assertEqual(loaded.size(), n)
                self.assertIsInstance(loaded, TriangularBitMatrix)
                # Check a few values if possible, or just rely on no crash
            finally:
                loaded.close()
        finally:
            matrix.close()

    def test_transposed_matrix(self):
        """Test that transposition flag is preserved."""
        n = 100
        matrix = TriangularBitMatrix.random(n, 0.1, seed=42)
        transposed = matrix.transpose()
        path = self.test_dir / "transposed.pycauset"
        
        try:
            self.assertTrue(transposed.is_transposed())
            pycauset.save(transposed, path)
            
            loaded = pycauset.load(path)
            try:
                self.assertTrue(loaded.is_transposed())
                self.assertEqual(loaded.size(), n)
            finally:
                loaded.close()
        finally:
            matrix.close()
            transposed.close()

    def test_float_matrix(self):
        """Test FloatMatrix storage."""
        n = 50
        matrix = FloatMatrix(n)
        # Fill diagonal
        for i in range(n):
            matrix.set(i, i, 3.14)
            
        # Verify before save
        self.assertAlmostEqual(matrix.get(0, 0), 3.14)
            
        path = self.test_dir / "float.pycauset"
        
        try:
            pycauset.save(matrix, path)
            
            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, FloatMatrix)
                self.assertAlmostEqual(loaded.get(0, 0), 3.14)
                self.assertAlmostEqual(loaded.get(n-1, n-1), 3.14)
                self.assertEqual(loaded.get(0, 1), 0.0)
            finally:
                loaded.close()
        finally:
            matrix.close()

    def test_integer_matrix(self):
        """Test IntegerMatrix storage."""
        n = 50
        matrix = IntegerMatrix(n)
        matrix.set(0, 0, 42)
        matrix.set(1, 2, -7)
        
        # Verify before save
        self.assertEqual(matrix.get(0, 0), 42)
        
        path = self.test_dir / "int.pycauset"
        
        try:
            pycauset.save(matrix, path)
            
            loaded = pycauset.load(path)
            try:
                self.assertIsInstance(loaded, IntegerMatrix)
                self.assertEqual(loaded.get(0, 0), 42)
                self.assertEqual(loaded.get(1, 2), -7)
            finally:
                loaded.close()
        finally:
            matrix.close()

    def test_causet_spacetime_metadata(self):
        """Test CausalSet with specific spacetime configuration."""
        # Use a Cylinder spacetime
        st = MinkowskiCylinder(dimension=2, height=10.0, circumference=5.0)
        c = CausalSet(n=100, spacetime=st, seed=999)
        path = self.test_dir / "cylinder.causet"
        
        try:
            c.save(path)
            
            # Verify metadata manually
            with zipfile.ZipFile(path, "r") as zf:
                with zf.open("metadata.json") as f:
                    meta = json.load(f)
                    self.assertEqual(meta["object_type"], "CausalSet")
                    self.assertEqual(meta["spacetime"]["type"], "MinkowskiCylinder")
                    self.assertEqual(meta["spacetime"]["args"]["dimension"], 2)
                    # Note: height/circumference might not be in args if not exposed by C++ getter, 
                    # but let's check what we put in causet.py
            
            # Load back
            loaded_c = CausalSet.load(path)
            try:
                self.assertEqual(loaded_c.n, 100)
                # Verify spacetime type if possible, or just that it loaded
            finally:
                loaded_c.C.close()
        finally:
            c.C.close()

    def test_overwrite(self):
        """Test overwriting an existing file."""
        n = 10
        matrix = IntegerMatrix(n)
        path = self.test_dir / "overwrite.pycauset"
        
        try:
            # First save
            pycauset.save(matrix, path)
            self.assertTrue(path.exists())
            
            # Second save
            pycauset.save(matrix, path)
            self.assertTrue(path.exists())
            
            # Load to verify not corrupted
            loaded = pycauset.load(path)
            loaded.close()
        finally:
            matrix.close()

    def test_nested_directory(self):
        """Test saving to a non-existent nested directory."""
        n = 10
        matrix = IntegerMatrix(n)
        path = self.test_dir / "subdir" / "nested" / "matrix.pycauset"
        
        try:
            pycauset.save(matrix, path)
            self.assertTrue(path.exists())
        finally:
            matrix.close()

    def test_unicode_path(self):
        """Test saving to a path with unicode characters."""
        n = 10
        matrix = IntegerMatrix(n)
        # "Fire" emoji and some non-ascii chars
        path = self.test_dir / "🔥_folder" / "mätrix.pycauset"
        
        try:
            pycauset.save(matrix, path)
            self.assertTrue(path.exists())
            
            loaded = pycauset.load(path)
            loaded.close()
        finally:
            matrix.close()

if __name__ == '__main__':
    unittest.main()
