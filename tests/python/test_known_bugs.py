"""Regression pins for known correctness bugs.

Each test runs the buggy path in an *isolated subprocess* (because some of
these trigger native heap corruption / access violations that would kill the
whole test process) and asserts the CORRECT behaviour.

Marked ``unittest.expectedFailure``: while broken they read as xfail; the
moment the underlying bug is fixed they report as an *unexpected success*,
which is the signal to remove the marker. Do NOT convert to ``skip``.
"""

import subprocess
import sys
import unittest

import pycauset._pycauset as native


def _run_py(code):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )


class TestKnownBugs(unittest.TestCase):
    @unittest.expectedFailure
    def test_solve_returns_correct_solution(self):
        # BUG: solve() returns a result that does NOT satisfy A @ x = b,
        # even though invert() and matmul are individually correct.
        code = (
            "import pycauset as pc, numpy as np\n"
            "rng = np.random.default_rng(0)\n"
            "a = rng.random((5, 5)); a += np.eye(5) * 5\n"
            "b = rng.random((5, 2))\n"
            "x = np.array(pc.solve(pc.matrix(a), pc.matrix(b)))\n"
            "print('OK' if np.allclose(a @ x, b, atol=1e-8) else 'WRONG')\n"
        )
        r = _run_py(code)
        self.assertIn("OK", r.stdout)

    @unittest.expectedFailure
    def test_lu_completes_and_reconstructs(self):
        # BUG: lu() raises MemoryError in result bookkeeping after computing
        # the factorization (get_backing_file() on the permutation matrix).
        code = (
            "import pycauset as pc, numpy as np\n"
            "a = np.random.default_rng(0).random((5, 5))\n"
            "p, l, u = pc.lu(pc.matrix(a))\n"
            "rec = np.array(p) @ np.array(l) @ np.array(u)\n"
            "print('OK' if np.allclose(rec, a, atol=1e-8) else 'WRONG')\n"
        )
        r = _run_py(code)
        self.assertIn("OK", r.stdout)

    @unittest.expectedFailure
    def test_triangular_bit_matrix_random_size(self):
        # BUG: TriangularBitMatrix.random(5) reports the wrong .size().
        tbm = native.TriangularBitMatrix.random(5, p=0.5)
        self.assertEqual(tbm.size(), 5)


if __name__ == "__main__":
    unittest.main()
