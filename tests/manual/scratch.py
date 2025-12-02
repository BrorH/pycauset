import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))
import pycauset as pc

c = pc.CausalSet(100)
print(c.C)