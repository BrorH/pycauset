from importlib import import_module
import random
import json
import zipfile
import shutil
import os
import struct
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
        # if path.suffix != ".causet":
        #     path = path.with_suffix(".causet")
            
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
            },
            "pycauset_version": "3.0",
            "object_type": "CausalSet",
            "matrix_type": "CAUSAL",
            "data_type": "BIT",
            "rows": self._matrix.size(),
            "cols": self._matrix.size(),
            "scalar": 1.0,
            "is_transposed": False
        }
        
        # Create ZIP
        # We need to handle both file-backed and memory-backed matrices.
        # copy_storage() creates a physical file copy regardless of the source type.
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        temp_raw = path.with_suffix(".raw_tmp")
        self._matrix.copy_storage(str(temp_raw))
        
        try:
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as zf:
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))
                zf.write(temp_raw, "data.bin")
        finally:
            # Clean up the temporary copy
            if temp_raw.exists():
                try:
                    temp_raw.unlink()
                except OSError:
                    pass

    @staticmethod
    def load(path: str | os.PathLike) -> 'CausalSet':
        """
        Load a CausalSet from a file.
        
        Args:
            path: Path to the file.
            
        Returns:
            CausalSet: The reconstructed object.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        with zipfile.ZipFile(path, 'r') as zf:
            # 1. Read Metadata
            with zf.open("metadata.json") as f:
                metadata = json.load(f)
            
            # 2. Reconstruct Spacetime
            st_info = metadata.get("spacetime", {})
            st_type = st_info.get("type", "MinkowskiDiamond")
            st_args = st_info.get("args", {"dimension": 2})
            
            if st_type == "MinkowskiDiamond":
                spacetime = _native.MinkowskiDiamond(st_args.get("dimension", 2))
            elif st_type == "MinkowskiCylinder":
                spacetime = _native.MinkowskiCylinder(
                    st_args.get("dimension", 2), 
                    st_args.get("height", 1.0), 
                    st_args.get("circumference", 1.0)
                )
            else:
                # Fallback
                spacetime = _native.MinkowskiDiamond(2)
                
            # 3. Find offset of data.bin
            try:
                info = zf.getinfo("data.bin")
                with open(path, "rb") as f:
                    f.seek(info.header_offset + 26)
                    n_len, e_len = struct.unpack("<HH", f.read(4))
                    data_offset = info.header_offset + 30 + n_len + e_len
            except KeyError:
                raise ValueError("Invalid file format: missing data.bin")

            # 4. Load Matrix directly from ZIP
            rows = metadata.get("rows", metadata.get("n"))
            seed = metadata.get("seed", 0)
            
            matrix = _native.TriangularBitMatrix(
                rows, 
                str(path), 
                data_offset, 
                seed, 
                1.0, 
                False
            )

            # 5. Create CausalSet
            return CausalSet(
                n=metadata.get("n"),
                spacetime=spacetime,
                seed=seed,
                matrix=matrix
            )

