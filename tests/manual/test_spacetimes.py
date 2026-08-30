import numpy as np
from pycauset import CausalSet, MinkowskiDiamond, MinkowskiCylinder, MinkowskiBox
from pycauset.vis import plot_embedding, plot_hasse

def test_spacetimes():
    print("Testing MinkowskiBox...")
    # Create a Box (Block)
    box = MinkowskiBox(2, 2.0, 1.0) # T=2, L=1
    c_box = CausalSet(n=100, spacetime=box, seed=42)

    print(f"Box created with N={c_box.n}")

    # Test Visualization
    print("Plotting Box Embedding...")
    fig = plot_embedding(c_box, title="Minkowski Box")

    print("Plotting Box Hasse...")
    fig_hasse = plot_hasse(c_box, title="Minkowski Box Hasse")

if __name__ == "__main__":
    try:
        test_spacetimes()
        print("All spacetime tests passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
