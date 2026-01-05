import pycauset as pc
import time
import os
import gc

# Set backing dir to a temp location
pc.set_backing_dir(".pycauset_leak_test")

CHUNK_SIZE = 1024 * 1024 * 1024 // 8  # 128M elements (double) = 1GB
CHUNKS = 10 # 10GB total

print(f"Starting Leak Test: {CHUNKS} chunks of 1GB each...")

try:
    for i in range(CHUNKS):
        print(f"Allocating Chunk {i+1}/{CHUNKS}...")
        
        # Create 1GB matrix
        # We force it to be file-backed by setting a low RAM threshold
        pc.set_memory_threshold(100 * 1024 * 1024) # 100MB
        
        M = pc.zeros((CHUNK_SIZE, 1), dtype="float64")
        
        # Touch data to force page-in
        M[0, 0] = 1.0
        M[CHUNK_SIZE-1, 0] = 1.0
        
        # Delete M to trigger cleanup
        del M
        gc.collect()
        
        # Sleep briefly to let OS catch up
        time.sleep(0.1)

    print("Test Passed (did not crash).")

except Exception as e:
    print(f"Test Failed: {e}")
    exit(1)
finally:
    # Cleanup
    try:
        import shutil
        if os.path.exists(".pycauset_leak_test"):
            shutil.rmtree(".pycauset_leak_test")
    except:
        pass
