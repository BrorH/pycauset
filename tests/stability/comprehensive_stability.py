import numpy as np
import pycauset
import time
import random
import os
import gc
import psutil

# Set process priority to high to catch race conditions? Maybe not needed.
process = psutil.Process()

def get_ram_usage_mb():
    return process.memory_info().rss / 1024 / 1024

def run_stability_check(duration_seconds=60, verify_frequency=10):
    print(f"=== PyCauset Comprehensive Stability Check ===")
    print(f"Duration: {duration_seconds}s")
    print(f"PID: {process.pid}")
    
    start_time = time.time()
    iter_count = 0
    errors = 0
    
    # 1. Base Setup
    N = 1500 # Dimensions (small enough for frequent ops, big enough to matter)
    np.random.seed(123)
    random.seed(123)
    
    # Keep some persistent objects to test lifetime management mixed with temporaries
    persistent_pool = []
    
    print("Initializing Pool...")
    for _ in range(5):
        mat_np = np.random.rand(N, N)
        mat_pc = pycauset.matrix(mat_np)
        persistent_pool.append({"pc": mat_pc, "np": mat_np})
        
    print(f"Initial RAM: {get_ram_usage_mb():.1f} MB")
    
    while time.time() - start_time < duration_seconds:
        iter_count += 1
        elapsed = time.time() - start_time
        
        # Action Choice
        action = random.choice([
            "arithmetic", "mat_mul", "ooc_pressure", "gc_hammer", "verify"
        ])
        
        # --- A. Mixed Arithmetic ---
        if action == "arithmetic":
            idx = random.randint(0, len(persistent_pool)-1)
            entry = persistent_pool[idx]
            
            op = random.choice(["add_scalar", "sub_matrix", "mul_scalar"])
            
            if op == "add_scalar":
                val = random.random()
                res_pc = entry["pc"] + val
                # Don't verify every time to save time
                if random.random() < 0.1:
                    res_np = entry["np"] + val
                    # Check
            
            elif op == "sub_matrix":
                idx2 = random.randint(0, len(persistent_pool)-1)
                other = persistent_pool[idx2]
                res_pc = entry["pc"] - other["pc"]
                
            elif op == "mul_scalar":
                val = random.random()
                res_pc = entry["pc"] * val
                
        # --- B. Matrix Multiplication ---
        elif action == "mat_mul":
            idx1 = random.randint(0, len(persistent_pool)-1)
            idx2 = random.randint(0, len(persistent_pool)-1)
            
            # Use small slice sometimes to test views? Not implemented deeply yet.
            # Just do full mul
            res_pc = persistent_pool[idx1]["pc"] @ persistent_pool[idx2]["pc"]
            
        # --- C. OOC Pressure ---
        elif action == "ooc_pressure":
            # Force low memory threshold for a moment
            limit = 100 * 1024 * 1024 # 100 MB
            pycauset.set_memory_threshold(limit)
            
            # Create a big temporary matrix
            big_N = 4000
            temp = pycauset.FloatMatrix(big_N, big_N)
            temp.fill(1.0)
            
            # Trigger io
            temp2 = temp * 2.0
            
            # Release
            del temp
            del temp2
            
            # Reset threshold
            pycauset.set_memory_threshold(8 * 1024 * 1024 * 1024)
            
        # --- D. GC Hammer ---
        elif action == "gc_hammer":
            gc.collect()
            
        # --- E. Verification Step ---
        elif action == "verify" or (iter_count % verify_frequency == 0):
            print(f"[t={elapsed:.1f}s Iter={iter_count} RAM={get_ram_usage_mb():.1f}MB] Verifying randomly...", end="")
            idx = random.randint(0, len(persistent_pool)-1)
            entry = persistent_pool[idx]
            
            # Perform a complex chain
            val = random.random()
            chain_pc = (entry["pc"] * val) + 1.0
            chain_np = (entry["np"] * val) + 1.0
            
            try:
                if hasattr(chain_pc, "to_numpy"):
                    res = chain_pc.to_numpy(allow_huge=True)
                else:
                    res = np.array(chain_pc)
                    
                if not np.allclose(chain_np, res, atol=1e-4, rtol=1e-4):
                    print(f" FAIL! Max Diff: {np.max(np.abs(chain_np - res))}")
                    errors += 1
                else:
                    print(f" OK.")
            except Exception as e:
                print(f" EXCEPTION: {e}")
                errors += 1

    print(f"\n=== Finished ===")
    print(f"Total Iterations: {iter_count}")
    print(f"Total Errors: {errors}")
    print(f"Final RAM: {get_ram_usage_mb():.1f} MB")
    
    if errors == 0:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
        exit(1)

if __name__ == "__main__":
    # Ensure backing dir exists
    pycauset.set_backing_dir(".pycauset_stability")
    run_stability_check(duration_seconds=30) # 30 seconds for interactive check, longer for CI
