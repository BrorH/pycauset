import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))
from pycauset import Causet, spacetime

from pycauset import CausalSet, MinkowskiDiamond
from pycauset.vis import plot_hasse, plot_causal_matrix

# Generate a small causal set
c = CausalSet(n=50, spacetime=MinkowskiDiamond(2), seed=42)

# Plot Hasse Diagram

from pycauset import CausalSet, MinkowskiDiamond
from pycauset.vis import plot_hasse, plot_causal_matrix

# Generate a small causal set
c = CausalSet(n=50, spacetime=MinkowskiCylinder(2), seed=42)

# Plot Hasse Diagram
fig_hasse = plot_hasse(c, title="Hasse Diagram")
fig_hasse.show()