"""Recover the spacetime dimension from the order alone (Myrheim-Meyer).

A causet sprinkled into a d-dimensional Minkowski diamond has a relation
fraction that depends on d. Myrheim-Meyer inverts that: nothing but the order
in, dimension estimate out.
"""

import pycauset as pc

for d in (2, 3, 4):
    st = pc.spacetime.MinkowskiDiamond(dimension=d)
    c = pc.causet(n=400, spacetime=st, seed=42)
    # c.relation_fraction() and c.myrheim_meyer_dimension() read the order only.
    print(f"built in {d}D -> estimate {c.myrheim_meyer_dimension():.2f}")
