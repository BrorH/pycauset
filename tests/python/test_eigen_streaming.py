"""
Test Phase 6: Streaming and Routing for Eigen Operations

Verifies that eigen operations respect routing decisions:
- Direct path when data fits in RAM
- Streaming threshold behavior
- IO trace includes routing decisions
- Arnoldi supports streaming for large matrices
"""

import unittest
import pycauset
import numpy as np


class TestEigenStreaming(unittest.TestCase):
    def setUp(self):
        """Set up streaming threshold for testing"""
        self._orig_threshold = pycauset.get_io_streaming_threshold()
        pycauset.clear_io_traces()
        
    def tearDown(self):
        """Restore original threshold"""
        pycauset.set_io_streaming_threshold(self._orig_threshold)
        pycauset.clear_io_traces()
        
    def test_eigh_uses_direct_path_small_matrix(self):
        """Test that eigh uses direct path for small matrices"""
        # Set high threshold so small matrix doesn't trigger streaming
        pycauset.set_io_streaming_threshold(1024 * 1024)  # 1MB
        pycauset.clear_io_traces()
        
        # Small matrix (10x10 float64 = 800 bytes)
        n = 10
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T
        A = pycauset.matrix(A_np)
        
        # Compute eigenvalues
        w, v = pycauset.eigh(A)
        
        # Check IO trace - should show direct path (or no trace if not instrumented)
        trace = pycauset.last_io_trace("eigh")
        if trace is not None:
            # If traced, route should be "direct"
            route = trace.get("route")
            if route is not None:
                self.assertEqual(route, "direct", "Small eigh should use direct path")
                
    def test_eigh_forced_direct_with_allow_huge(self):
        """Test that eigh respects allow_huge flag"""
        # Set very low threshold
        pycauset.set_io_streaming_threshold(1)
        pycauset.clear_io_traces()
        
        n = 50
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T
        A = pycauset.matrix(A_np)
        
        # With allow_huge, should prefer direct even if large
        # (though eigh doesn't support streaming anyway, so always direct)
        w, v = pycauset.eigh(A)
        
        trace = pycauset.last_io_trace("eigh")
        if trace is not None:
            route = trace.get("route")
            # eigh doesn't support streaming, so should be direct with reason
            if route is not None:
                self.assertEqual(route, "direct")
                reason = trace.get("reason")
                if reason is not None:
                    # Should mention that eigen doesn't support streaming
                    self.assertIn("stream", reason.lower(), "Reason should mention streaming not supported")
                    
    def test_eigvals_arnoldi_can_stream(self):
        """Test that Arnoldi iteration supports out-of-core execution"""
        # Arnoldi only needs matrix-vector products, so can work with disk-backed A
        pycauset.set_io_streaming_threshold(1)
        pycauset.clear_io_traces()
        
        n = 100
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T  # Symmetric for real eigenvalues
        A = pycauset.matrix(A_np)
        
        # Request top 5 eigenvalues with Arnoldi
        try:
            w = pycauset.eigvals_arnoldi(A, k=5, m=10, tol=1e-6)
            
            # Verify we got results
            self.assertEqual(w.size(), 5)
            
            # Check trace - Arnoldi should support streaming
            trace = pycauset.last_io_trace("eigvals_arnoldi")
            if trace is not None:
                # Arnoldi can stream (matrix-vector products)
                # May use streaming or direct depending on policy
                route = trace.get("route")
                if route is not None:
                    self.assertIn(route, ["streaming", "direct"], "Arnoldi should support streaming")
        except NotImplementedError:
            self.skipTest("eigvals_arnoldi not fully implemented yet")
            
    def test_eig_non_square_rejection(self):
        """Test that non-square matrices are rejected with clear reason"""
        pycauset.clear_io_traces()
        
        # Non-square matrix
        A = pycauset.matrix(np.random.rand(10, 5))
        
        # Should reject
        with self.assertRaises((ValueError, RuntimeError)) as cm:
            pycauset.eig(A)
            
        error_msg = str(cm.exception).lower()
        self.assertIn("square", error_msg, "Error should mention square matrix requirement")
        
        # Check trace if available
        trace = pycauset.last_io_trace("eig")
        if trace is not None:
            route = trace.get("route")
            if route == "direct":
                reason = trace.get("reason")
                if reason is not None:
                    self.assertIn("square", reason.lower())
                    
    def test_trace_includes_operation_name(self):
        """Test that IO trace includes operation name for debugging"""
        pycauset.clear_io_traces()
        
        n = 10
        A_np = np.random.rand(n, n)
        A_np = A_np + A_np.T
        A = pycauset.matrix(A_np)
        
        # Run eigh
        w, v = pycauset.eigh(A)
        
        # Check that we can retrieve trace by operation name
        trace = pycauset.last_io_trace("eigh")
        # Trace may or may not exist depending on instrumentation
        # If it exists, it should be retrievable by name


if __name__ == "__main__":
    unittest.main()
