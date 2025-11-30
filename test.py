import pycauset as pc



# a = pc.CausalMatrix(5)
# pc.save(a, "atest")

a = pc.load("atest")

k = pc.compute_k(a, 2.1)
print(2*k)