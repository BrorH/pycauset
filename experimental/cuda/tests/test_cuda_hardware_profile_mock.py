import os
import unittest

import pycauset


class TestCudaHardwareProfileMock(unittest.TestCase):
    def test_cuda_hardware_profile_mock_env(self):
        if not pycauset.cuda.is_available():
            self.skipTest("CUDA not available")

        env_key = "PYCAUSET_TEST_CUDA_PROFILE"
        old_value = os.environ.get(env_key)
        try:
            os.environ[env_key] = (
                "device_id=0;device_name=Mock GPU;cc_major=8;cc_minor=6;"
                "pci_bandwidth_gbps=12.5;sgemm_gflops=1234.0;dgemm_gflops=456.0"
            )

            profile = pycauset.cuda.benchmark(force=True)
            self.assertIsNotNone(profile)
            self.assertEqual(profile["device_id"], 0)
            self.assertEqual(profile["device_name"], "Mock GPU")
            self.assertEqual(profile["cc_major"], 8)
            self.assertEqual(profile["cc_minor"], 6)
            self.assertAlmostEqual(profile["pci_bandwidth_gbps"], 12.5, places=4)
            self.assertAlmostEqual(profile["sgemm_gflops"], 1234.0, places=4)
            self.assertAlmostEqual(profile["dgemm_gflops"], 456.0, places=4)
        finally:
            if old_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = old_value
