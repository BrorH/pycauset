"""R2_STREAM: elementwise ops stream out-of-core under a forced threshold.

The StreamingManager already tiles `add`/`subtract`/`elementwise_multiply`/
`elementwise_divide` (via `StreamingManager::elementwise`), but the SRP handoff
table still marked them "naive". This test pins the correctness of that tiled
path with a forced IO-streaming threshold, so the support matrix can honestly
claim streaming-enabled for these ops.
"""

import unittest

import numpy as np

import pycauset


class TestOutOfCoreElementwise(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_threshold = pycauset.get_io_streaming_threshold()

    def tearDown(self) -> None:
        pycauset.set_io_streaming_threshold(self._orig_threshold)

    def _check(self, np_fn, pc_fn) -> None:
        rng = np.random.default_rng(0)
        n = 256
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        got = np.asarray(pc_fn(pycauset.matrix(a), pycauset.matrix(b)))
        want = np_fn(a, b)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    def test_add_subtract_multiply_divide_stream_out_of_core(self) -> None:
        pycauset.set_io_streaming_threshold(64)  # force the tiled/streaming route

        self._check(lambda a, b: a + b, lambda A, B: A + B)
        self._check(lambda a, b: a - b, lambda A, B: A - B)
        self._check(lambda a, b: a * b, lambda A, B: A * B)
        self._check(lambda a, b: a / b, lambda A, B: A / B)


if __name__ == "__main__":
    unittest.main()
