from importlib import import_module
import random
import os
from pathlib import Path
from typing import Optional, Sequence, List, Union

# Import the native extension relative to this package
try:
    _native = import_module("._pycauset", package=__package__)
except ImportError:
    # Try absolute import as fallback
    import pycauset._pycauset as _native

try:  # NumPy is optional at runtime
    import numpy as _np
except ImportError:  # pragma: no cover - exercised when numpy is absent
    _np = None

class CausalSet:
    def __init__(self, n: int = None, density: float = None, spacetime=None, seed: Union[int, str] = None, matrix=None):
        """
        Initialize a CausalSet.
        
        This can be done in two ways:
        1. Sprinkling: Provide 'n' (or 'density') and optionally 'spacetime' and 'seed'.
        2. Loading: Provide 'n', 'spacetime', 'seed', and 'matrix' (used by load()).

        Args:
            n (int, optional): Number of elements.
            density (float, optional): Density of sprinkling. If provided, n is calculated as Poisson(density * volume).
            spacetime (CausalSpacetime, optional): The spacetime to sprinkle into. 
                                             Defaults to 2D MinkowskiDiamond.
            seed (int | str, optional): Random seed. Can be an integer or a string. Defaults to random.
            matrix (TriangularBitMatrix, optional): Pre-existing matrix. If provided, sprinkling is skipped.
        """
        # --- Spacetime Setup ---
        if spacetime is None:
            self._spacetime = _native.MinkowskiDiamond(2)
        else:
            self._spacetime = spacetime
            
        # --- Seed Setup ---
        if seed is None:
            self._seed = random.randint(0, 2**63 - 1)
        elif isinstance(seed, int):
            self._seed = seed
        else:
            # Support strings or other hashable objects as seeds
            # We use Python's random module to deterministically map the seed to an integer
            rng = random.Random(seed)
            self._seed = rng.randint(0, 2**63 - 1)
            
        # --- N / Density Setup ---
        if n is not None:
            self._n = int(n)
        elif density is not None:
            # Calculate n from density
            if _np is None:
                raise ImportError("NumPy is required for density-based sprinkling (Poisson distribution).")
            
            # Seed the numpy RNG with our seed to ensure reproducibility of N
            rng = _np.random.default_rng(self._seed)
            volume = self._spacetime.volume()
            self._n = rng.poisson(density * volume)
        else:
            raise ValueError("Must provide either 'n' or 'density'.")

        # --- Matrix Generation / Assignment ---
        if matrix is not None:
            if hasattr(matrix, "rows") and hasattr(matrix, "cols"):
                if matrix.rows() != self._n or matrix.cols() != self._n:
                    raise ValueError(
                        f"Provided matrix shape ({matrix.rows()}, {matrix.cols()}) does not match n ({self._n})."
                    )
            else:
                if matrix.size() != self._n:
                    raise ValueError(f"Provided matrix size ({matrix.size()}) does not match n ({self._n}).")
            self._matrix = matrix
        else:
            # Generate the matrix immediately using the stateless sprinkler
            self._matrix = _native.sprinkle(self._spacetime, self._n, self._seed)

    @property
    def causal_matrix(self):
        """The causal matrix (TriangularBitMatrix) representing the causal relations."""
        return self._matrix

    @property
    def C(self):
        """Alias for causal_matrix."""
        return self._matrix
    
    @property
    def n(self):
        """The number of elements in the causal set."""
        return self._n

    @property
    def N(self):
        """Alias for n."""
        return self._n

    @property
    def density(self):
        """
        The density of the sprinkling.
        Calculated as N / Volume.
        """
        return self._n / self._spacetime.volume()

    @property
    def rho(self):
        """Alias for density."""
        return self.density
        
    @property
    def spacetime(self):
        return self._spacetime
        
    def __repr__(self):
        return f"<CausalSet n={self._n} spacetime={self._spacetime}>"
    
    def __len__(self):
        return self._n

    def coordinates(self, indices: Optional[Sequence[int]] = None, force: bool = False):
        """
        Retrieve spacetime coordinates for specific elements.
        
        Args:
            indices: List of element indices to retrieve. If None, retrieves all (subject to safety limits).
            force: If True, bypasses safety limits for large sets.
            
        Returns:
            numpy.ndarray: Array of shape (K, D) where K is number of indices.
        """
        if indices is None:
            if self.n > 100000 and not force:
                raise UserWarning(
                    f"CausalSet has {self.n} elements. Retrieving all coordinates is expensive. "
                    "Use 'indices' to select a subset or set 'force=True' to proceed anyway."
                )
            indices = list(range(self.n))
        
        # Ensure indices are a list of ints
        indices = [int(i) for i in indices]
        
        coords = _native.make_coordinates(self._spacetime, self._n, self._seed, indices)
        
        if _np:
            return _np.array(coords)
        return coords

    def save(self, path: str | os.PathLike):
        """Save the CausalSet to the single-file `.pycauset` container."""
        import pycauset as _pycauset
        _pycauset.save(self, Path(path))

    @staticmethod
    def load(path: str | os.PathLike) -> 'CausalSet':
        """Load a CausalSet from the single-file `.pycauset` container."""
        import pycauset as _pycauset
        obj = _pycauset.load(Path(path))
        if not isinstance(obj, CausalSet):
            raise ValueError("file did not contain a CausalSet")
        return obj

