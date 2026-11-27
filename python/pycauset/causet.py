"""CausalSet: a finite causal order (a causal set), for R1 the sprinkled Minkowski diamond.

A `CausalSet` is a bit (dtype "bool") `TriangularMatrix` of causal relations plus provenance
(a spacetime and a seed). It carries the order methods (`links`, `past` / `future`,
`interval`, chains / antichains, layers), the Myrheim-Meyer dimension estimate and three
lightweight plots. Fields and the rest of the spacetime library are parked in
`experimental/physics`.

**Natural ordering (a core invariant).** Elements are labelled by time (ascending coordinate
0), so the causal matrix is strictly upper triangular: ``i < j`` whenever ``i`` precedes
``j``. That is the storage invariant the bit-packed engine relies on; `sprinkle` and the
`synthetic` generators obey it.
"""


import os
import random
from importlib import import_module
from pathlib import Path
from typing import Optional, Sequence, Union

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

from .spacetime import MinkowskiDiamond as _DefaultDiamond


def validate_causal_matrix(matrix, *, context: str = "causal matrix") -> None:
    """Validate that a causal matrix is a strict partial order.

    A causal set's order must be:
    * **reflexive-free**, no element is its own cause (zero diagonal);
    * **antisymmetric**, never both ``i ≺ j`` and ``j ≺ i``;
    * **transitive**, the matrix is the *closure*, not just the links (every
      length-2 path must be a direct relation).

    Raises ``ValueError`` with an actionable message on the first violation.
    """
    if _np is None:
        raise ImportError("NumPy is required for causal-matrix validation.")

    A = _np.asarray(matrix, dtype=bool)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"{context}: causal matrix must be square, got shape {A.shape}"
        )

    if A.diagonal().any():
        raise ValueError(
            f"{context}: causal matrix must be reflexive-free (zero diagonal)"
        )

    if (A & A.T).any():
        raise ValueError(
            f"{context}: causal matrix must be antisymmetric "
            "(never both i \u227a j and j \u227a i)"
        )

    # Transitivity: a length-2 path (i -> k -> j) requires the direct relation i -> j. The
    # engine counts paths exactly (bit matrices multiply in int32); a uint8 NumPy product
    # wrapped 256 paths to 0 and passed intransitive orders (F-107).
    import pycauset as _pc

    M = matrix if isinstance(matrix, _native.MatrixBase) else _pc.matrix(A)
    path2 = _np.asarray(M @ M) > 0
    bad = path2 & ~A
    if bad.any():
        i, j = _np.argwhere(bad)[0]
        raise ValueError(
            f"{context}: causal matrix is not transitive "
            f"(a length-2 path {int(i)} \u2192 \u22ef \u2192 {int(j)} has no direct relation)"
        )


class CausalSet:
    def __init__(
        self,
        n: int = None,
        density: float = None,
        spacetime=None,
        seed: Union[int, str] = None,
        matrix=None,
        validate: bool = True,
    ):
        """
        Initialize a CausalSet.

        Two ways:
        1. Sprinkling: provide 'n' (or 'density') and optionally 'spacetime' and 'seed'.
        2. From an order: provide 'matrix' (a causal matrix) with 'n' (load() does this).

        Natural ordering (core invariant): elements are labelled by time (ascending
        coordinate 0), so the causal matrix is strictly upper-triangular — ``i ≺ j``
        only when ``i < j``. Every causet produced here obeys this; a manually
        supplied ``matrix`` must already be in that form (only the upper triangle is
        stored).

        Args:
            n (int, optional): Number of elements.
            density (float, optional): Density of sprinkling. If provided, n is calculated as Poisson(density * volume).
            spacetime (pycauset.spacetime.MinkowskiDiamond, optional): the region to
                sprinkle into. Defaults to the 2D diamond.
            seed (int | str, optional): Random seed. Defaults to random.
            matrix (TriangularMatrix of dtype "bool", optional): Pre-existing matrix. If provided, sprinkling is skipped.
            validate (bool): When True (default), a supplied `matrix` is validated as a
                strict partial order (reflexive-free, antisymmetric, transitive).
        """
        # --- Spacetime Setup ---
        self._spacetime = _DefaultDiamond(2) if spacetime is None else spacetime

        # --- Seed Setup ---
        if seed is None:
            self._seed = random.randint(0, 2**63 - 1)
        elif isinstance(seed, int):
            self._seed = seed
        else:
            rng = random.Random(seed)
            self._seed = rng.randint(0, 2**63 - 1)

        # --- N / Density Setup ---
        if n is not None:
            self._n = int(n)
        elif density is not None:
            if _np is None:
                raise ImportError("NumPy is required for density-based sprinkling (Poisson distribution).")
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
            if validate:
                validate_causal_matrix(matrix, context="CausalSet(matrix=...)")
        else:
            self._matrix = _native._sprinkle(self._spacetime._native_spacetime(), self._n, self._seed)

    @property
    def causal_matrix(self):
        """The causal matrix (bit TriangularMatrix) representing the causal relations."""
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
        """The density of the sprinkling, calculated as N / Volume."""
        return self._n / self._spacetime.volume()

    @property
    def rho(self):
        """Alias for density."""
        return self.density

    @property
    def spacetime(self):
        return self._spacetime

    def validate(self) -> None:
        """Validate that the causal order is a strict partial order.

        Raises ``ValueError`` if the matrix is not reflexive-free, antisymmetric,
        and transitive.
        """
        validate_causal_matrix(self._matrix, context="CausalSet")

    # --- causal structure (R2_STRUCT: "a causet is just a poset") ---

    def links(self):
        """Transitive reduction (Hasse / link matrix): ``L = C & ~(C@C)``."""
        # Native bit kernel on the (validated, transitively closed) causal matrix. The former
        # NumPy uint8 product wrapped path counts of 256 to 0 and reported false links (F-053).
        L = _native._transitive_reduction(self._matrix)
        return _np.asarray(L, dtype=bool)

    def past(self, x):
        """Indices ``i`` with ``i \u227a x``."""
        C = _np.asarray(self._matrix, dtype=bool)
        return _np.flatnonzero(C[:, int(x)])

    def future(self, x):
        """Indices ``j`` with ``x \u227a j``."""
        C = _np.asarray(self._matrix, dtype=bool)
        return _np.flatnonzero(C[int(x), :])

    def interval(self, x, y):
        """Alexandrov interval ``I(x, y) = future(x) \u2229 past(y)``."""
        return _np.intersect1d(self.future(x), self.past(y))

    def is_chain(self, elements) -> bool:
        """True if every pair in `elements` is comparable."""
        elements = list(elements)
        C = _np.asarray(self._matrix, dtype=bool)
        for a in range(len(elements)):
            for b in range(a + 1, len(elements)):
                i, j = int(elements[a]), int(elements[b])
                if not (C[i, j] or C[j, i]):
                    return False
        return True

    def is_antichain(self, elements) -> bool:
        """True if no pair in `elements` is comparable."""
        elements = list(elements)
        C = _np.asarray(self._matrix, dtype=bool)
        for a in range(len(elements)):
            for b in range(a + 1, len(elements)):
                i, j = int(elements[a]), int(elements[b])
                if C[i, j] or C[j, i]:
                    return False
        return True

    def longest_chain(self):
        """A longest causal chain (indices); its length is the poset's height."""
        if self._n == 0:
            return _np.array([], dtype=int)
        C = _np.asarray(self._matrix, dtype=bool)
        n = self._n
        dp = _np.ones(n, dtype=int)
        prev = _np.full(n, -1, dtype=int)
        for i in range(n):
            preds = _np.flatnonzero(C[:i, i])
            if preds.size:
                j = int(preds[dp[preds].argmax()])
                dp[i] = 1 + dp[j]
                prev[i] = j
        i = int(dp.argmax())
        chain = [i]
        while prev[i] != -1:
            i = int(prev[i])
            chain.append(i)
        return _np.array(chain[::-1], dtype=int)

    def layers(self):
        """Ranked layering: ``layers[k]`` = elements whose longest past-chain has length ``k+1``."""
        if self._n == 0:
            return []
        C = _np.asarray(self._matrix, dtype=bool)
        n = self._n
        rank = _np.ones(n, dtype=int)
        for i in range(n):
            preds = _np.flatnonzero(C[:i, i])
            if preds.size:
                rank[i] = 1 + int(rank[preds].max())
        return [_np.flatnonzero(rank == k + 1) for k in range(int(rank.max()))]

    # --- dimension estimators (R2_DIM) ---

    def relation_fraction(self) -> float:
        """Fraction of element pairs that are causally related: ``R / C(n, 2)``.

        This is a dimension-dependent statistic (≈ 1 in 0+1, 1/4 in 1+1, and
        smaller in higher dimensions) and the basis of the Myrheim\u2013Meyer
        dimension estimator.
        """
        if self._n < 2:
            return 0.0
        C = _np.asarray(self._matrix, dtype=bool)
        R = int(C.sum())
        return R / (self._n * (self._n - 1) / 2)

    def myrheim_meyer_dimension(self, dmin: float = 1.0, dmax: float = 8.0) -> float:
        """Myrheim\u2013Meyer dimension estimate (inverts the relation fraction).

        Uses ``f(d) = \u0393(d+1) \u0393(d/2) / (2 \u0393(3d/2))``, the fraction of
        comparable pairs in d-dimensional Minkowski (f(1)=1, f(2)=1/2, f(4)=1/10) -
        and bisects on the measured relation fraction. Unbiased for a sprinkled Minkowski
        diamond in any dimension (statistical error ~ 1/sqrt(n)).
        """
        import math

        f = self.relation_fraction()
        lo, hi = float(dmin), float(dmax)
        for _ in range(200):
            mid = (lo + hi) / 2.0
            fmid = math.gamma(mid + 1) * math.gamma(mid / 2) / (2.0 * math.gamma(3 * mid / 2))
            if fmid > f:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # --- plots (pycauset.vis; lightweight, lazy Plotly import) ---

    def plot_embedding(self, **kwargs):
        """Scatter plot of the elements at their coordinates. Returns a Plotly Figure."""
        from .vis import plot_embedding
        return plot_embedding(self, **kwargs)

    def plot_hasse(self, **kwargs):
        """The Hasse diagram: elements at their coordinates, lines along the links."""
        from .vis import plot_hasse
        return plot_hasse(self, **kwargs)

    def plot_causal_matrix(self, **kwargs):
        """The causal matrix as a heatmap."""
        from .vis import plot_causal_matrix
        return plot_causal_matrix(self, **kwargs)

    def __repr__(self):
        return f"<CausalSet n={self._n} spacetime={self._spacetime}>"

    def __len__(self):
        return self._n

    def coordinates(self, indices: Optional[Sequence[int]] = None, force: bool = False):
        """
        Retrieve spacetime coordinates for specific elements.

        Coordinates are regenerated from the ``(spacetime, seed)`` provenance (the same RNG
        stream as the sprinkle), time-labelled like the matrix.

        Args:
            indices: List of element indices to retrieve. If None, retrieves all (subject to safety limits).
            force: If True, bypasses safety limits for large sets.

        Returns:
            numpy.ndarray: Array of shape (K, D) where K is number of indices.
        """
        if indices is None and self.n > 100000 and not force:
            raise UserWarning(
                f"CausalSet has {self.n} elements. Retrieving all coordinates is expensive. "
                "Use 'indices' to select a subset or set 'force=True' to proceed anyway."
            )

        if indices is None:
            indices = list(range(self.n))

        indices = [int(i) for i in indices]
        coords = _native._make_coordinates(self._spacetime._native_spacetime(), self._n, self._seed, indices)

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
