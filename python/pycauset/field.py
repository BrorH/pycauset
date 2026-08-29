"""Fields on causal sets — the R2 field core.

The model is ``Field -> CorrelatedField`` (R2_FIELD):

* ``pc.field("scalar", mass=…)`` is a set-independent `Field` (the field content).
* ``phi.on(causet)`` returns a `CorrelatedField` — the field together with its
  Green's functions and vacuum two-point function on that causet.

The propagators follow R2_KRD:

* ``K_R = aC (I - baC)\u207b\u00b9``  (retarded Green's function)
* ``K_A = K_R\u1d40``                (advanced Green's function)
* ``i\u0394 = K_R - K_A``            (Pauli\u2013Jordan commutator function, Hermitian)

The Sorkin\u2013Johnston vacuum (R2_SJ, the flagship) is the positive-eigenvalue part
of ``i\u0394``, exposed via ``.wightman()``.
"""

from __future__ import annotations

import sys
import types
from typing import Optional, Tuple

import numpy as np

from .causet import CausalSet


def _scalar_coeffs(causet: CausalSet, mass: float) -> Tuple[float, float]:
    """Derive the ``(a, b)`` coefficients for a scalar on a causet's spacetime.

    Delegates to ``Spacetime.scalar_coeffs(mass, density)`` (R2_COEFFS): the built-in
    Minkowski spacetimes implement the known 2D/4D table, everything else raises —
    coefficients are never guessed and never name-sniffed.
    """
    try:
        rho = causet.density
    except (ValueError, AttributeError):
        rho = None

    if rho is None:
        raise ValueError(
            "Cannot compute field coefficients: CausalSet density is unknown. "
            "Ensure the CausalSet was created with density information or provide 'a' and 'b' manually."
        )

    return causet.spacetime.scalar_coeffs(mass, rho)


class Field:
    """Set-independent field content (R2_FIELD).

    ``kind`` is the species (``"scalar"`` for now), ``mass`` the field mass,
    ``spin`` and ``scheme`` reserved for future species.
    """

    def __init__(self, *, kind: str = "scalar", mass: float = 0.0, spin: int = 0, scheme=None):
        if kind != "scalar":
            raise NotImplementedError(
                f"field kind {kind!r} is not implemented (only 'scalar')"
            )
        self.kind = kind
        self.mass = float(mass)
        self.spin = int(spin)
        self.scheme = scheme

    def on(self, background):
        """Apply the field to a background.

        ``phi.on(causet)`` returns a discrete `CorrelatedField`; ``phi.on(spacetime)``
        returns a continuum `ContinuumCorrelatedField` (R2_CMVP).
        """
        if self.kind != "scalar":
            raise NotImplementedError(self.kind)  # pragma: no cover
        if isinstance(background, CausalSet):
            return CorrelatedField(background, mass=self.mass)
        return ContinuumCorrelatedField(background, mass=self.mass)


class CorrelatedField:
    """A field together with its Green's functions and vacuum two-point function
    on a causal set (R2_FIELD / R2_KRD / R2_SJ). Returns dense NumPy matrices."""

    def __init__(self, causet: CausalSet, mass: float = 0.0):
        self._causet = causet
        self._mass = float(mass)

    @property
    def causet(self) -> CausalSet:
        return self._causet

    @property
    def mass(self) -> float:
        return self._mass

    def _coeffs(self) -> Tuple[float, float]:
        return _scalar_coeffs(self._causet, self._mass)

    def retarded(self, a: Optional[float] = None, b: Optional[float] = None) -> np.ndarray:
        """Retarded Green's function ``K_R = aC (I - baC)\u207b\u00b9`` (dense n\u00d7n)."""
        from . import compute_k

        C = self._causet.C
        if a is None or b is None:
            ca, cb = self._coeffs()
            a = ca if a is None else a
            b = cb if b is None else b

        if abs(b) < 1e-15:
            return a * np.asarray(C, dtype=float)

        alpha = -1.0 / (a * b)
        X = compute_k(C, alpha)  # C @ inv(alpha*I + C)
        return (-1.0 / b) * np.asarray(X, dtype=float)

    def advanced(self, a: Optional[float] = None, b: Optional[float] = None) -> np.ndarray:
        """Advanced Green's function ``K_A = K_R\u1d40``."""
        return self.retarded(a, b).T

    def pauli_jordan(self) -> np.ndarray:
        """Pauli\u2013Jordan function ``i\u0394 = K_R - K_A`` (Hermitian)."""
        K = self.retarded()
        return 1j * (K - K.T)

    def wightman(self) -> np.ndarray:
        """Sorkin\u2013Johnston vacuum Wightman two-point function.

        ``W`` is the positive-eigenvalue part of ``i\u0394``: diagonalize the
        Hermitian ``i\u0394`` and keep the non-negative eigenvalues.
        """
        iDelta = self.pauli_jordan()
        evals, evecs = np.linalg.eigh(iDelta)
        return (evecs * np.clip(evals, 0.0, None)) @ evecs.conj().T

    def correlator(self) -> np.ndarray:
        """Vacuum two-point function ``\u27e8\u03c6\u03c6\u27e9 = W`` (free field)."""
        return self.wightman()

    def propagator(self, a: Optional[float] = None, b: Optional[float] = None) -> np.ndarray:
        """Alias for `retarded()`."""
        return self.retarded(a, b)

    def state(self, config=None) -> State:
        """Build a `State` (a field configuration) on top of this correlated field.

        ``config`` is a real vector of length ``n`` (the classical field values at
        each element); ``None`` gives the vacuum (zero configuration).
        """
        n = self._causet.n
        if config is None:
            config = np.zeros(n)
        return State(self, config)

    def entanglement_entropy(self, region, convention="sorkin_yazdi") -> float:
        """Sorkin\u2013Yazdi entanglement entropy of a region (R2_ENT).

        Computes the entanglement entropy of a region (a subset of element indices)
        from the region-restricted SJ Wightman matrix ``W_A``.

        **Conventions (documented):**

        * ``"sorkin_yazdi"`` (default) — the zero-point ``1/2`` convention:
          ``S = tr[(W_A + I) ln(W_A + I) \u2212 W_A ln W_A]``, with ``0 ln 0 = 0``.
          Well-defined for the SJ Wightman (``W_A \u2265 0``).
        * ``"symplectic"`` — the literal symplectic-eigenvalue form
          ``S = tr[(W_A + 1/2) ln(W_A + 1/2) \u2212 (W_A \u2212 1/2) ln(W_A \u2212 1/2)]``,
          which assumes ``W_A \u2265 1/2`` (a Wightman already in the covariance
          convention); raises `ValueError` otherwise.

        Parameters:
            region: iterable of element indices.
            convention: ``"sorkin_yazdi"`` or ``"symplectic"``.

        Returns:
            float: the entanglement entropy (``\u2265 0``).
        """
        W = self.wightman()
        idx = [int(i) for i in region]
        W_A = W[np.ix_(idx, idx)]
        evals = np.linalg.eigvalsh(W_A).astype(float)

        if convention == "sorkin_yazdi":
            s = np.zeros_like(evals)
            nz = evals > 0.0
            s[nz] = (evals[nz] + 1.0) * np.log(evals[nz] + 1.0) - evals[nz] * np.log(evals[nz])
            return float(np.sum(s))

        if convention == "symplectic":
            if np.any(evals < 0.5):
                raise ValueError(
                    "convention='symplectic' requires the restricted Wightman to have "
                    "eigenvalues >= 1/2 (the covariance convention); the SJ Wightman "
                    "(positive part of i\u0394) has eigenvalues >= 0. Use "
                    "convention='sorkin_yazdi' (the 1/2 zero-point convention) instead."
                )
            s = (evals + 0.5) * np.log(evals + 0.5) - (evals - 0.5) * np.log(evals - 0.5)
            return float(np.sum(s))

        raise ValueError(
            f"unknown convention {convention!r}; use 'sorkin_yazdi' or 'symplectic'"
        )


class State:
    """A specific excitation of the vacuum (R2_FIELD: Field \u2192 CorrelatedField \u2192 State).

    A `State` is a coherent/classical field configuration ``phi`` (a real vector of
    length n) over the causet, carrying the vacuum two-point function ``W``. Its
    expectation values are (for a Gaussian/coherent state):

    * ``\u27e8\u03c6\u27e9 = phi``
    * ``\u27e8\u03c6\u03c6\u27e9 = phi phi\u1d40 + W``
    * ``\u27e8\u03c6\u00b2\u27e9 = diag(W) + phi\u00b2``
    """

    def __init__(self, correlated_field: CorrelatedField, config: np.ndarray):
        self._cf = correlated_field
        self._phi = np.asarray(config, dtype=float)
        if self._phi.shape != (correlated_field.causet.n,):
            raise ValueError(
                f"state config must have length {correlated_field.causet.n}, "
                f"got shape {self._phi.shape}"
            )
        self._W = correlated_field.wightman()

    @property
    def correlated_field(self) -> CorrelatedField:
        return self._cf

    def field(self) -> np.ndarray:
        """``\u27e8\u03c6\u27e9`` — the mean field configuration."""
        return self._phi.copy()

    def two_point(self) -> np.ndarray:
        """``\u27e8\u03c6\u03c6\u27e9 = phi phi\u1d40 + W``."""
        return np.outer(self._phi, self._phi) + self._W

    def field_variance(self) -> np.ndarray:
        """``\u27e8\u03c6\u00b2\u27e9 = diag(W) + phi\u00b2`` (per-element fluctuation)."""
        return np.real(np.diag(self._W)) + self._phi ** 2


class ContinuumCorrelatedField:
    """Continuum comparison (R2_CMVP): closed-form Green's functions on flat Minkowski.

    Covers the **massless 1+1** case exactly (the state-independent ``i\u0394`` is
    ``(i/2) sgn(\u0394t) \u03b8(\u03c3)``). Massive and higher-dimensional closed forms
    (Bessel functions) are flagged for R2_CONV.
    """

    def __init__(self, spacetime, mass: float = 0.0):
        name = type(spacetime).__name__
        if "Minkowski" not in name:
            raise NotImplementedError(
                f"continuum comparison for spacetime {name!r} is not implemented"
            )
        self._spacetime = spacetime
        self._mass = float(mass)
        self._d = int(spacetime.dimension())

    def _sigma(self, x, y):
        # Convention (R2_CONV): `dt = y[0] - x[0]` is positive when y is in the
        # FUTURE of x, so the discrete matrix convention ``K_R[i,j]`` ("from the
        # past element i to the future element j") maps to ``retarded(x, y)``.
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dt = y[0] - x[0]
        dx = x[1:] - y[1:]
        return dt * dt - float(np.dot(dx, dx)), dt

    def retarded(self, x, y) -> float:
        """Continuum retarded Green's function ``G_R(x, y)``."""
        sigma, dt = self._sigma(x, y)
        if self._mass != 0.0:
            raise NotImplementedError(
                "massive continuum Green's functions require Bessel functions (R2_CONV)"
            )
        if self._d == 2:
            return 0.5 if (dt > 0 and sigma > 0) else 0.0
        raise NotImplementedError(
            f"massless continuum G_R in d={self._d} is distributional (\u03b4 on the lightcone)"
        )

    def advanced(self, x, y) -> float:
        """Continuum advanced Green's function ``G_A(x, y) = G_R(y, x)``."""
        return self.retarded(y, x)

    def pauli_jordan(self, x, y) -> complex:
        """Continuum Pauli\u2013Jordan ``i\u0394(x, y) = i(G_R - G_A)`` (massless 1+1 exact)."""
        sigma, dt = self._sigma(x, y)
        if self._mass != 0.0 or self._d != 2:
            raise NotImplementedError(
                "continuum i\u0394 is exact only for the massless 1+1 case (R2_CONV)"
            )
        return 0.5j * np.sign(dt) * (sigma > 0)

    def wightman(self, x, y):
        """Continuum Wightman two-point function (R2_CONV: log/Bessel convention)."""
        raise NotImplementedError(
            "continuum Wightman closed form is R2_CONV (log/Bessel convention pinning)"
        )

    def at(self, coords, which: str = "pauli_jordan"):
        """Sample a kernel at the ``(n, d)`` coordinates, returning an ``(n, n)`` matrix."""
        coords = np.asarray(coords, dtype=float)
        n = coords.shape[0]
        fn = {"retarded": self.retarded, "advanced": self.advanced,
              "pauli_jordan": self.pauli_jordan, "wightman": self.wightman}[which]
        is_complex = which == "pauli_jordan"
        out = np.zeros((n, n), dtype=complex if is_complex else float)
        for i in range(n):
            for j in range(n):
                out[i, j] = fn(coords[i], coords[j])
        return out


def field(kind: str = "scalar", *, mass: float = 0.0, spin: int = 0, scheme=None) -> Field:
    """String-factory sugar: ``pc.field("scalar", mass=…)`` returns a `Field`.

    Unknown kinds raise (never guessed).
    """
    return Field(kind=kind, mass=mass, spin=spin, scheme=scheme)


# --- back-compat: the R1 ScalarField (keeps the native-matrix return types) ---


class ScalarField:
    """A massive scalar field defined on a Causal Set (R1 API, kept for back-compat).

    The retarded propagator is ``K_R = aC (I - baC)\u207b\u00b9``; the coefficients
    ``a, b`` derive from the spacetime dimension, sprinkling density, and mass.
    """

    def __init__(
        self,
        causet: CausalSet | None = None,
        mass: float = 0.0,
        *,
        n: int | None = None,
        density: float | None = None,
        spacetime=None,
        seed=None,
        matrix=None,
    ):
        if causet is None:
            causet = CausalSet(n=n, density=density, spacetime=spacetime, seed=seed, matrix=matrix)
        self._causet = causet
        self._mass = float(mass)
        self._cached_propagator = None

    @property
    def causet(self) -> CausalSet:
        return self._causet

    @property
    def mass(self) -> float:
        return self._mass

    def _get_coeffs(self) -> Tuple[float, float]:
        return _scalar_coeffs(self._causet, self._mass)

    def compute_retarded_propagator(self, a: Optional[float] = None, b: Optional[float] = None):
        if self._cached_propagator is not None and a is None and b is None:
            return self._cached_propagator

        if a is None or b is None:
            calc_a, calc_b = self._get_coeffs()
            if a is None:
                a = calc_a
            if b is None:
                b = calc_b

        C = self._causet.C

        if abs(b) < 1e-15:
            result = a * C
        else:
            alpha_eff = -1.0 / (a * b)
            from . import compute_k
            X = compute_k(C, alpha_eff)
            result = (-1.0 / b) * X

        if a is None and b is None:
            self._cached_propagator = result

        return result

    def propagator(self, a: Optional[float] = None, b: Optional[float] = None):
        """Alias for compute_retarded_propagator."""
        return self.compute_retarded_propagator(a, b)

    def pauli_jordan(self):
        """Pauli-Jordan function ``i\u0394 = K_R - K_R\u1d40``, stored antisymmetrically."""
        K = self.compute_retarded_propagator()
        from . import AntiSymmetricFloat64Matrix
        Delta = AntiSymmetricFloat64Matrix.from_triangular(K)
        Delta.set_scalar(1j)
        return Delta


# Make `pycauset.field` callable (R2 string factory) while remaining a module, so
# both `pc.field("scalar", mass=…)` and `from pycauset.field import ScalarField`
# keep working.
class _FieldModule(types.ModuleType):
    def __call__(self, kind: str = "scalar", **kwargs) -> Field:
        return field(kind, **kwargs)


sys.modules[__name__].__class__ = _FieldModule

__all__ = ["Field", "CorrelatedField", "ContinuumCorrelatedField", "ScalarField", "State", "field"]
