import unittest

import pycauset


class TestStreamingManager(unittest.TestCase):
    def setUp(self):
        self._orig_threshold = pycauset.get_io_streaming_threshold()
        pycauset.set_io_streaming_threshold(64)
        pycauset.clear_io_traces()

    def tearDown(self):
        pycauset.set_io_streaming_threshold(self._orig_threshold)
        pycauset.clear_io_traces()

    def test_matmul_descriptor_sets_plan_and_tile(self):
        a = pycauset.zeros((4, 4), dtype=pycauset.float64)
        b = pycauset.zeros((4, 4), dtype=pycauset.float64)

        pycauset.matmul(a, b)

        trace = pycauset.last_io_trace("matmul")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get("route"), "streaming")
        self.assertEqual(trace.get("queue_depth"), 3)
        self.assertEqual(trace.get("tile_shape"), (2, 2))
        plan_section = trace.get("plan", {})
        self.assertEqual(plan_section.get("access_pattern"), "blocked_rowcol")

    def test_invert_guard_flips_non_square_to_direct(self):
        manager = getattr(pycauset, "_STREAMING_MANAGER", None)
        self.assertIsNotNone(manager)

        pycauset.set_io_streaming_threshold(16)
        pycauset.clear_io_traces()

        class Rectangular:
            def __init__(self, rows, cols):
                self._rows = rows
                self._cols = cols

            def rows(self):
                return self._rows

            def cols(self):
                return self._cols

        rect = Rectangular(2, 3)

        plan = manager.plan("invert", [rect], allow_huge=False)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.get("route"), "direct")
        self.assertEqual(plan.get("reason"), "non_square")
        self.assertEqual(plan.get("queue_depth"), 0)
        self.assertIsNone(plan.get("tile_shape"))

        plan_events = plan.get("events", []) if isinstance(plan, dict) else []
        self.assertTrue(
            any(evt.get("type") == "plan" and evt.get("reason") == "non_square" for evt in plan_events)
        )


if __name__ == "__main__":
    unittest.main()
