import numpy as np
import plotly.graph_objects as go
from pycauset import CausalSet, MinkowskiDiamond, MinkowskiCylinder, MinkowskiBox
from pycauset.vis import plot_embedding, plot_hasse
from pycauset.field import ScalarField

def test_spacetimes():
    print("Testing MinkowskiBox...")
    # Create a Box (Block)
    box = MinkowskiBox(2, 2.0, 1.0) # T=2, L=1
    c_box = CausalSet(n=100, spacetime=box, seed=42)
    
    print(f"Box created with N={c_box.n}")
    
    # Test Visualization
    print("Plotting Box Embedding...")
    fig = plot_embedding(c_box, title="Minkowski Box")
    # fig.show()
    
    print("Plotting Box Hasse...")
    fig_hasse = plot_hasse(c_box, title="Minkowski Box Hasse")
    
    # Test Field Coefficients
    print("Testing Field Coefficients...")
    field = ScalarField(c_box, mass=1.0)
    # This should call _box_coeffs
    # We need to access the private method or just check if propagator() runs without error
    # But propagator() does heavy calculation. Let's check _get_coeffs directly if possible
    # or just trust that if it didn't raise NotImplementedError, it worked.
    
    try:
        coeffs = field._get_coeffs()
        print(f"Coefficients for Box (2D): {coeffs}")
        assert coeffs[0] == 0.5
    except Exception as e:
        print(f"Failed to get coefficients: {e}")
        raise

    print("Testing MinkowskiDiamond (4D)...")
    diamond_4d = MinkowskiDiamond(4)
    c_4d = CausalSet(n=50, spacetime=diamond_4d, seed=42)
    field_4d = ScalarField(c_4d, mass=1.0)
    try:
        coeffs_4d = field_4d._get_coeffs()
        print(f"Coefficients for Diamond (4D): {coeffs_4d}")
        # Check 'a' value approx
        rho = c_4d.density
        expected_a = np.sqrt(rho) / (2 * np.pi * np.sqrt(6))
        assert np.isclose(coeffs_4d[0], expected_a)
    except Exception as e:
        print(f"Failed to get 4D coefficients: {e}")
        raise

if __name__ == "__main__":
    try:
        test_spacetimes()
        print("All spacetime tests passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
