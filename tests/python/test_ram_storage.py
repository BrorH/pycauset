import pycauset
import numpy as np
import os
# import pytest

def test_ram_backed_small_object():
    # Set threshold to 1MB (10^6 bytes)
    pycauset.set_memory_threshold(1000000)
    
    # Create a small matrix (100x100 bits is very small)
    # 100*100 bits = 10000 bits = 1250 bytes << 1MB
    m = pycauset.TriangularBitMatrix(100)
    
    # Check if it's temporary
    assert m.is_temporary
    
    # Check backing file - should be empty or special indicator if we exposed it, 
    # but get_backing_file() returns ":memory:" for RAM objects now?
    # Let's check what get_backing_file() returns.
    print(f"Backing file: {m.get_backing_file()}")
    assert m.get_backing_file() == ":memory:"
    
    # Verify functionality
    m.set(0, 1, True)
    assert m.get(0, 1) == True
    assert m.get(0, 2) == False
    
    # Verify numpy interop
    arr = np.array(m)
    assert arr[0, 1] == 1
    assert arr.shape == (100, 100)

def test_disk_backed_large_object():
    # Set threshold to very small (1 byte)
    pycauset.set_memory_threshold(1)
    
    # Create a matrix larger than 1 byte
    m = pycauset.TriangularBitMatrix(100)
    
    print(f"Backing file: {m.get_backing_file()}")
    assert m.get_backing_file() != ":memory:"
    assert os.path.exists(m.get_backing_file())

def test_save_ram_object():
    pycauset.set_memory_threshold(1000000)
    m = pycauset.TriangularBitMatrix(100)
    m.set(0, 1, True)
    
    save_path = "test_ram_save.pycauset"
    if os.path.exists(save_path):
        os.remove(save_path)
        
    try:
        pycauset.save(m, save_path)
        assert os.path.exists(save_path)
        
        # Load it back
        m2 = pycauset.load(save_path)
        assert m2.get(0, 1) == True
        assert m2.size() == 100
        # Loaded object should be disk-backed (unless we load into RAM? load() maps the file provided)
        # assert m2.get_backing_file() == os.path.abspath(save_path)
        m2.close() # Close to release file lock
        
    finally:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except PermissionError:
                print("Could not remove file, likely still open.")

if __name__ == "__main__":
    # Manual run if pytest not available
    try:
        test_ram_backed_small_object()
        print("test_ram_backed_small_object PASSED")
        test_disk_backed_large_object()
        print("test_disk_backed_large_object PASSED")
        test_save_ram_object()
        print("test_save_ram_object PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
