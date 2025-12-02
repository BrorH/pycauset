from importlib import import_module
import random

# Import the native extension relative to this package
try:
    _native = import_module("._pycauset", package=__package__)
except ImportError:
    # Try absolute import as fallback
    import pycauset._pycauset as _native

class CausalSet:
    def __init__(self, n: int, spacetime=None, seed: int = None):
        """
        Initialize a CausalSet by sprinkling points into a spacetime.

        Args:
            n (int): Number of elements.
            spacetime (CausalSpacetime, optional): The spacetime to sprinkle into. 
                                             Defaults to 2D MinkowskiDiamond.
            seed (int, optional): Random seed. Defaults to random.
        """
        self._n = n
        if spacetime is None:
            self._spacetime = _native.MinkowskiDiamond(2)
        else:
            self._spacetime = spacetime
            
        if seed is None:
            self._seed = random.randint(0, 2**63 - 1)
        else:
            self._seed = seed
            
        # Generate the matrix immediately using the stateless sprinkler
        self._matrix = _native.sprinkle(self._spacetime, self._n, self._seed)

    @property
    def CausalMatrix(self):
        """
        The causal matrix (TriangularBitMatrix) representing the causal relations.
        This is the primary product of the sprinkling process.
        """
        return self._matrix

    @property
    def C(self):
        """Alias for CausalMatrix."""
        return self._matrix
    
    @property
    def n(self):
        return self._n
        
    @property
    def spacetime(self):
        return self._spacetime
        
    def __repr__(self):
        return f"<CausalSet n={self._n} spacetime={self._spacetime}>"
