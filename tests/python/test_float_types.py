import unittest
import os
import numpy as np
from pycauset import Matrix, Float16Matrix, Float32Matrix, save, load

class TestFloatTypes(unittest.TestCase):
    def test_00_float16_matrix(self):
        print("\nTesting Float16Matrix...")
        n = 4
        m = Float16Matrix(n)
        m.set(0, 0, 3.5)
        self.assertAlmostEqual(float(m.get(0, 0)), 3.5, places=3)

        # NumPy roundtrip (dtype + values)
        m_np = np.array(m)
        self.assertEqual(m_np.dtype, np.float16)
        self.assertAlmostEqual(float(m_np[0, 0]), 3.5, places=3)

        # Matmul should work end-to-end (float32 compute, float16 storage)
        m2 = Float16Matrix(n)
        m2.set(0, 0, 2.0)
        res = m.multiply(m2)
        self.assertAlmostEqual(float(res.get(0, 0)), 7.0, places=2)

        # Persistence roundtrip
        tmp_path = os.path.abspath("tmp_test_float16.zip")
        try:
            save(m, tmp_path)
            m_loaded = load(tmp_path)
            self.assertEqual(m_loaded.__class__.__name__, "Float16Matrix")
            self.assertAlmostEqual(float(m_loaded.get(0, 0)), 3.5, places=3)
            m_loaded.close()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        m.close()
        m2.close()
        res.close()

    def test_01_float32_matrix(self):
        print("\nTesting Float32Matrix...")
        n = 100
        m = Float32Matrix(n)
        m.set(0, 0, 3.14159)
        val = m.get(0, 0)
        self.assertAlmostEqual(val, 3.14159, places=5)
        
        # Test multiplication
        m2 = Float32Matrix(n)
        m2.set(0, 0, 2.0)
        
        res = m.multiply(m2)
        # res[0,0] = row 0 of m * col 0 of m2.
        # row 0 of m has 3.14 at index 0, 0 elsewhere.
        # col 0 of m2 has 2.0 at index 0, 0 elsewhere.
        # So res[0,0] should be 3.14 * 2.0 = 6.28318
        
        self.assertAlmostEqual(res.get(0, 0), 6.28318, places=4)
        
        m.close()
        m2.close()
        res.close()

    def test_03_smart_defaults(self):
        print("\nTesting Smart Defaults...")
        # Small matrix -> Float64 (standard FloatMatrix)
        m_small = Matrix(100)
        self.assertEqual(m_small.__class__.__name__, "FloatMatrix")
        m_small.close()
        
        # Medium matrix -> Float32Matrix
        # We mock the size check by forcing it or just trusting the logic.
        # Since we can't easily allocate 10k matrix in a quick test without disk usage,
        # we can check if force_precision works.
        
        m_f32 = Matrix(100, force_precision="float32")
        self.assertEqual(m_f32.__class__.__name__, "Float32Matrix")
        m_f32.close()

    def test_04_matrix_from_numpy_float16(self):
        print("\nTesting Matrix(np.float16 ndarray)...")
        a = np.zeros((4, 4), dtype=np.float16)
        a[0, 0] = np.float16(1.25)
        m = Matrix(a)
        self.assertEqual(m.__class__.__name__, "Float16Matrix")
        self.assertAlmostEqual(float(m.get(0, 0)), 1.25, places=3)
        m.close()

if __name__ == '__main__':
    unittest.main()
