import unittest
import pycauset as pc
import os
import struct
import time

class TestSafety(unittest.TestCase):
    def test_corrupt_header(self):
        print("Testing corrupt header load...")
        # Create valid file
        M = pc.zeros((10, 10), dtype="float64")
        pc.save(M, "test_corrupt.pycauset")
        
        # Corrupt Magic
        with open("test_corrupt.pycauset", "r+b") as f:
            f.write(b"BADMAGIC")
            
        # Load should fail
        with self.assertRaises(Exception): # ValueError or RuntimeError
            pc.load("test_corrupt.pycauset")
            
        # Cleanup
        try:
            os.remove("test_corrupt.pycauset")
        except:
            pass

    def test_spill_integrity(self):
        print("Testing spill integrity (Simple Header)...")
        # Force spill
        pc.set_memory_threshold(1024) # 1KB
        M = pc.zeros((100, 100), dtype="float64") # 80KB
        
        # This should spill to a .tmp file with Simple Header.
        # If offset logic is correct, M[0,0] should be 0.0 (payload), not "PYCAUSET" (header).
        
        val = M[0, 0]
        self.assertEqual(val, 0.0)
        
        # Write something and read back
        M[0, 0] = 123.456
        self.assertEqual(M[0, 0], 123.456)
        
        # Check if we can find the file and verify header
        # This is tricky because filename is internal.
        # But we can scan .pycauset dir.
        
    def test_gpu_fallback_logic(self):
        # We can't easily force GPU failure, but we can ensure the code runs without crashing
        # even if we try to use GPU features.
        pass

if __name__ == "__main__":
    unittest.main()
