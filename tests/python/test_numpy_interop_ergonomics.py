import pycauset
import numpy as np
import pytest

def test_mixed_op_add():
    """Test A(pycauset) + B(numpy) and B(numpy) + A(pycauset)."""
    A = pycauset.FloatMatrix(5, 5)
    A.fill(10.0)
    B = np.full((5, 5), 2.0, dtype=np.float64)

    # A + B (PyCauset + NumPy)
    print("  A + B...", flush=True)
    C1 = A + B
    print(f"  A + B result type: {type(C1)}", flush=True)
    assert isinstance(C1, pycauset.FloatMatrix)
    assert C1[0, 0] == 12.0

    # B + A (NumPy + PyCauset)
    print("  B + A...", flush=True)
    C2 = B + A
    print(f"  B + A result type: {type(C2)}", flush=True)
    assert isinstance(C2, pycauset.FloatMatrix)
    assert C2[0, 0] == 12.0

def test_mixed_op_sub():
    A = pycauset.FloatMatrix(5, 5)
    A.fill(10.0)
    B = np.full((5, 5), 2.0, dtype=np.float64)

    # A - B
    C1 = A - B
    assert C1[0, 0] == 8.0

    # B - A
    C2 = B - A
    # 2.0 - 10.0 = -8.0
    assert C2[0, 0] == -8.0

def test_mixed_op_mul_scalar():
    A = pycauset.FloatMatrix(5, 5)
    A.fill(10.0)
    s = np.float64(2.0)

    # A * scalar
    C3 = A * s
    assert isinstance(C3, pycauset.FloatMatrix)
    assert C3[0, 0] == 20.0

    # scalar * A
    C4 = s * A
    assert isinstance(C4, pycauset.FloatMatrix)
    assert C4[0, 0] == 20.0

def test_mixed_op_div():
    A = pycauset.FloatMatrix(5, 5)
    A.fill(10.0)
    B = np.full((5, 5), 2.0, dtype=np.float64)

    # A / B -> 5.0
    C1 = A / B
    assert C1[0, 0] == 5.0

    # B / A -> 0.2
    C2 = B / A
    assert C2[0, 0] == 0.2

if __name__ == "__main__":
    print("Testing add...", flush=True)
    test_mixed_op_add()
    print("Testing sub...", flush=True)
    test_mixed_op_sub()
    print("Testing mul...", flush=True)
    test_mixed_op_mul_scalar()
    print("Testing div...", flush=True)
    test_mixed_op_div()
    print("All mixed op tests passed!", flush=True)
