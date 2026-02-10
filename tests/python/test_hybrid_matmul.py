import pytest
import pycauset as pc
import numpy as np

# Helper to check if "cuda" submodule exists and gpu available
def has_gpu():
    if not hasattr(pc, 'cuda'):
        return False
    return pc.cuda.is_available()

@pytest.mark.skipif(not has_gpu(), reason="GPU required for hybrid test")
def test_hybrid_matmul_correctness():
    print("\n[Test] Setting up Hybrid Execution (50% CPU, 50% GPU)...")
    
    # Setup
    n = 2048 
    
    # Force Hybrid
    cfg = pc.cuda.RoutingConfig()
    cfg.use_gpu = True
    cfg.allow_hybrid = True
    cfg.hybrid_cpu_ratio = 0.5 
    cfg.debug_routing = True
    # n*n = 4M elements. Default threshold is 260k. Should trigger.
    
    pc.cuda.set_config(cfg)
    
    try:
        # Data
        print(f"[Test] Generating {n}x{n} matrices...")
        A_np = np.random.rand(n, n).astype(np.float64)
        B_np = np.random.rand(n, n).astype(np.float64)
        
        A = pc.Matrix(A_np)
        B = pc.Matrix(B_np)
        
        # Hybrid Run
        print("[Test] Running A @ B...")
        C = A @ B
        
        # Verify
        print("[Test] Verifying results...")
        C_np = C.numpy()
        Expected = A_np @ B_np
        
        np.testing.assert_allclose(C_np, Expected, rtol=1e-5, atol=1e-5)
        print("[Test] Verification Passed!")
        
    finally:
        # Reset config
        cfg_reset = pc.cuda.RoutingConfig() 
        pc.cuda.set_config(cfg_reset)
        print("[Test] Config reset.")

if __name__ == "__main__":
    if has_gpu():
        test_hybrid_matmul_correctness()
    else:
        print("Skipping: No GPU")
