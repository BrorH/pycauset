import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))
import pycauset as pc

C = pc.CausalMatrix(1e5)
C.save("1e3_C.pycauset")
print(C)
# C = pc.load("1e3_C.pycauset")
