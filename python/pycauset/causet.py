from importlib import import_module
import random
import json
import zipfile
import shutil
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
            if matrix.size() != self._n:
                raise ValueError(f"Provided matrix size ({matrix.size()}) does not match n ({self._n}).")
            self._matrix = matrix
        else:
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

    def compute_k(self, a: float = 1.0):
        """
        DEPRECATED: Use pycauset.field.ScalarField(c).propagator() instead.
        
        Compute the K-matrix (retarded propagator) for this causal set
        assuming a massless scalar field with manual coefficient 'a'.
        
        Args:
            a (float): The non-locality scale parameter. Defaults to 1.0.
            
        Returns:
            TriangularFloatMatrix: The computed K matrix.
        """
        import warnings
        warnings.warn(
            "CausalSet.compute_k is deprecated. Use pycauset.field.ScalarField(c).propagator() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Local import to avoid circular dependency
        from . import compute_k
        return compute_k(self.C, a)

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
        """
        Save the CausalSet to a .causet file (ZIP archive).
        
        The archive contains:
        - metadata.json: Parameters (n, seed, spacetime config).
        - matrix.bin: The raw binary backing file of the causal matrix.
        
        Args:
            path: Destination path.
        """
        path = Path(path)
        if path.suffix != ".causet":
            path = path.with_suffix(".causet")
            
        # Prepare metadata
        st_type = self._spacetime.__class__.__name__
        st_args = {}
        
        if st_type == "MinkowskiDiamond":
            st_args["dimension"] = self._spacetime.dimension()
        elif st_type == "MinkowskiCylinder":
            st_args["dimension"] = self._spacetime.dimension()
            st_args["height"] = self._spacetime.height
            st_args["circumference"] = self._spacetime.circumference
            
        metadata = {
            "n": self._n,
            "seed": self._seed,
            "spacetime": {
                "type": st_type,
                "args": st_args
            }
        }
        
        # Create ZIP
        # We need to handle both file-backed and memory-backed matrices.
        # copy_storage() creates a physical file copy regardless of the source type.
        temp_matrix_path = self._matrix.copy_storage("")
        
        try:
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))
                zf.write(temp_matrix_path, "matrix.bin")
        finally:
            # Clean up the temporary copy
            if os.path.exists(temp_matrix_path):
                try:
                    os.remove(temp_matrix_path)
                except OSError:
                    pass

    @staticmethod
    def load(path: str | os.PathLike) -> 'CausalSet':
        """
        Load a CausalSet from a .causet file.
        
        Args:
            path: Path to the .causet file.
            
        Returns:
            CausalSet: The reconstructed object.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        with zipfile.ZipFile(path, 'r') as zf:
            # 1. Read Metadata
            metadata = json.loads(zf.read("metadata.json"))
            
            # 2. Reconstruct Spacetime
            st_info = metadata["spacetime"]
            st_type = st_info["type"]
            st_args = st_info["args"]
            
            if st_type == "MinkowskiDiamond":
                spacetime = _native.MinkowskiDiamond(st_args["dimension"])
            elif st_type == "MinkowskiCylinder":
                spacetime = _native.MinkowskiCylinder(
                    st_args["dimension"], 
                    st_args["height"], 
                    st_args["circumference"]
                )
            else:
                # Fallback or error for unknown types
                raise ValueError(f"Unknown spacetime type: {st_type}")
                
            # 3. Extract Matrix
            # We need to extract matrix.bin to a temporary location that persists
            # so the TriangularBitMatrix can map it.
            # We use the _storage module's mechanism if possible, or just a temp file.
            from . import _storage_root
            from ._storage import set_temporary_file
            
            # Generate a unique path in our storage directory
            import uuid
            temp_name = f"loaded_{uuid.uuid4().hex}.bin"
            temp_path = _storage_root() / temp_name
            
            with zf.open("matrix.bin") as source, open(temp_path, "wb") as target:
                shutil.copyfileobj(source, target)
                
            # Register for cleanup by marking it as temporary in the header
            set_temporary_file(temp_path, True)
            
            # 4. Load Matrix
            # We use _native.load() because it correctly handles opening existing files
            # without overwriting them (unlike the constructor which might default to new).
            matrix = _native.load(str(temp_path))
            
            if not isinstance(matrix, _native.TriangularBitMatrix):
                 # If it's not the right type, something is wrong with the file or logic
                 raise TypeError(f"Loaded matrix is not a TriangularBitMatrix, got {type(matrix)}")

            # 5. Create CausalSet
            return CausalSet(
                n=metadata["n"], # Use n from metadata, though matrix.size() should match
                spacetime=spacetime,
                seed=metadata["seed"],
                matrix=matrix
            )
