import pycauset as pc
import numpy as np



a = pc.CausalMatrix(10000,saveas="bigone")


k = pc.compute_k(a, 2.1,saveas="kbigone")

print(k)