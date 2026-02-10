import pytest
import pycauset as pc
import pycauset.cuda as cuda
import numpy as np

def has_gpu():
    if not hasattr(cuda, 'is_available'):
        return False
    return cuda.is_available()

@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestHybridCorrectnessSuite:
    
    @pytest.fixture(autouse=True)
    def setup_hybrid(self):
        # Configure Hybrid for all tests in suite
        cfg = cuda.RoutingConfig()
        cfg.use_gpu = True
        cfg.allow_hybrid = True
        cfg.hybrid_cpu_ratio = 0.25 # Aggressive split
        cfg.gpu_threshold_elements = 100 # Low threshold to force usage
        cfg.debug_routing = True
        cuda.set_config(cfg)
        yield
        # Teardown
        cuda.set_config(cuda.RoutingConfig())

    @pytest.mark.parametrize("n", [512, 1024, 2048])
    def test_matmul_square(self, n):
        idx = np.random.randint(0, 1000)
        print(f"Testing N={n}")
        
        # Use simple floats to avoid precision drift issues in super-strict tests,
        # but PyCauset does float64 by default for numpy inputs usually.
        # Let's stick to float32 for speed/tolerance unless hardware is A100.
        dtype = np.float32
        
        A_np = np.random.rand(n, n).astype(dtype)
        B_np = np.random.rand(n, n).astype(dtype)
        
        A = pc.Matrix(A_np)
        B = pc.Matrix(B_np)
        
        C = A * B
        
        C_np = C.numpy()
        Expected = A_np @ B_np
        
        # Check history to confirm Hybrid was actually used
        history = cuda.get_history()
        # Find the matmul op
        valid_ops = [h for h in history if h.op_code == 0] # MatMul=0
        if valid_ops:
            last = valid_ops[-1]
            assert "Hybrid" in last.details, f"Expected Hybrid execution, got: {last.reason} / {last.details}"
        
        np.testing.assert_allclose(C_np, Expected, rtol=1e-4, atol=1e-4)

    def test_matmul_rectangular_split(self):
        # Test a case where rows don't divide cleanly
        m, k, n = 1050, 512, 512 
        
        A_np = np.random.rand(m, k).astype(np.float32)
        B_np = np.random.rand(k, n).astype(np.float32)
        
        A = pc.Matrix(A_np)
        B = pc.Matrix(B_np)
        
        C = A @ B
        
        np.testing.assert_allclose(C.numpy(), A_np @ B_np, rtol=1e-4, atol=1e-4)

    def test_matmul_fallback_behavior(self):
        # Force a case where Hybrid is enabled but Input is too small for threshold
        cfg = cuda.get_config()
        cfg.gpu_threshold_elements = 10_000_000 # Huge
        cuda.set_config(cfg)
        
        n = 128
        A = pc.Matrix(n, n)
        B = pc.Matrix(n, n)
        C = A * B
        
        history = cuda.get_history()
        last = history[-1]
        assert not last.chosen_gpu
        assert "Threshold" in last.reason or "Threshold" in last.details
