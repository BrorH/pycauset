import numpy as np
import plotly.graph_objects as go
from pycauset import CausalSet, MinkowskiDiamond, MinkowskiCylinder
from pycauset.vis import plot_hasse, plot_causal_matrix

def test_vis():
    print("Creating CausalSet...")
    # Create a small diamond
    diamond = MinkowskiDiamond(2)
    cset = CausalSet(n=50, spacetime=diamond, seed=42)
    
    print("Testing plot_hasse (2D)...")
    fig_hasse = plot_hasse(cset, title="Test Hasse")
    # fig_hasse.show() # Cannot show in this env, but creating it is the test
    print("plot_hasse (2D) created successfully.")

    print("Testing plot_causal_matrix...")
    fig_matrix = plot_causal_matrix(cset, title="Test Matrix")
    print("plot_causal_matrix created successfully.")
    
    # Test 3D Hasse (Cylinder)
    print("Creating Cylinder CausalSet...")
    cylinder = MinkowskiCylinder(2, 2.0, 2.0)
    cset_cyl = CausalSet(n=50, spacetime=cylinder, seed=42)
    
    print("Testing plot_hasse (3D Cylinder)...")
    fig_hasse_3d = plot_hasse(cset_cyl, title="Test Hasse 3D")
    print("plot_hasse (3D) created successfully.")

if __name__ == "__main__":
    try:
        test_vis()
        print("All visualization tests passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
