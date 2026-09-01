"""Recover the spacetime dimension from the order alone (Myrheim-Meyer).

A causet sprinkled into a d-dimensional Minkowski diamond has a relation
fraction that depends on d. The Myrheim-Meyer estimator inverts that: feed it
nothing but the causal order and it gives back the dimension.
"""

import pycauset as pc

for d in (2, 3, 4):
    st = pc.spacetime.MinkowskiDiamond(dimension=d)
    c = pc.causet(n=400, spacetime=st, seed=42)
    frac = c.relation_fraction()
    est = c.myrheim_meyer_dimension()
    print(f"built in {d}D -> relation fraction {frac:.3f}, estimate {est:.2f}")
