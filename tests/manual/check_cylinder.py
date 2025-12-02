import pycauset
from pycauset import spacetime

try:
    c = spacetime.MinkowskiCylinder(2, 10.0, 5.0)
    print(f"Height: {c.height}")
    print(f"Circumference: {c.circumference}")
except Exception as e:
    print(e)
