"""Spacetime: the R2 extension seam.

`Spacetime` is the Python ABC that custom spacetimes implement, and the built-in
Minkowski spacetimes conform to. A first-class **signature** ``(t, s) =
(timelike, spacelike)`` is the only structural assumption; there is no hidden
Lorentzian guess.

Contract (R2_ABC, frozen early):

* ``dimension()``  , total ``d = t + s`` (index 0 is time for Lorentzian).
* ``volume()``     , total mass of the sampling measure (``0 < volume() < inf``).
* ``sample(rng, n)``- ``(n, d)`` array, uniform w.r.t. that measure, a pure
                      function of the injected RNG.
* ``is_causal(u,v)``- the strict, transitive partial order (the closure, not the
                      links). Meaningful only for Lorentzian ``t == 1``.

Optional hooks: ``is_causal_batch`` (Rung 2 fast path), ``scalar_coeffs``
(authored field coefficients, never guessed), ``to_embedding`` and ``boundary``
(presentation only). ``@spacetime.register("name")`` gives a spacetime a
persistable name.
"""

from __future__ import annotations

import abc
import itertools
import math
from typing import Optional, Tuple

import numpy as np

# The native (C++) spacetime classes are kept reachable for back-compat, but the
# canonical Minkowski spacetimes below are pure-Python `Spacetime` subclasses so
# the sprinkler can time-order points (fixing the R1 non-transitive order bug).
try:
    from . import _pycauset as _native  # noqa: F401  (back-compat accessor)
except ImportError:  # pragma: no cover
    _native = None

__all__ = [
    "Spacetime",
    "register",
    "create",
    "export_python",
    "get_registry",
    "RestrictedSpacetime",
    "TransformedSpacetime",
    "ConformalSpacetime",
    "PeriodicSpacetime",
    "MinkowskiDiamond",
    "MinkowskiCylinder",
    "MinkowskiBox",
    "DeSitter",
    "AntiDeSitter",
    "FLRW",
    "Schwarzschild",
]


class Spacetime(abc.ABC):
    """Continuum region + measure + causal order, the public extension seam.

    Subclasses implement ``dimension()``, ``volume()``, ``sample(rng, n)``, and
    ``is_causal(u, v)``. ``signature`` defaults to Lorentzian ``(1, d-1)``, a
    documented default, not an inference; override by declaring a class attribute
    ``signature = (t, s)``.
    """

    @property
    def signature(self) -> Tuple[int, int]:
        """``(t, s) = (timelike, spacelike)``; default Lorentzian ``(1, d-1)``.

        A subclass that declares ``signature = (t, s)`` as a class attribute
        shadows this property, so ``self.signature`` resolves to that tuple.
        """
        return (1, self.dimension() - 1)

    @abc.abstractmethod
    def dimension(self) -> int:
        """Total spacetime dimension ``d = t + s`` (index 0 is time)."""

    @abc.abstractmethod
    def volume(self) -> float:
        """Total mass of the sampling measure; ``0 < volume() < inf``."""

    @abc.abstractmethod
    def sample(self, rng, n):
        """Draw ``n`` points as an ``(n, d)`` array, uniform w.r.t. the measure
        whose mass is ``volume()``. Must be a pure function of ``rng``."""

    def is_causal(self, u, v) -> bool:
        """Strict partial order (the transitive closure), element-wise.

        A causal order exists only for a Lorentzian signature (``t == 1``). The
        base implementation raises for any other signature rather than guessing.
        """
        t, _ = self.signature
        if t != 1:
            raise NotImplementedError(
                f"`is_causal` is undefined for signature {self.signature}: a "
                "causal order exists only for a Lorentzian signature (t == 1). "
                "Euclidean spacetimes are point processes; multi-time spacetimes "
                "must supply their own 'future' convention."
            )
        raise NotImplementedError(
            f"{type(self).__name__} must implement is_causal(u, v)."
        )

    # --- optional (Rung 2 + presentation + physics) ---

    def is_causal_batch(self, coords):
        """``(n, n)`` boolean causal matrix (upper-triangular). Optional."""
        raise NotImplementedError

    def scalar_coeffs(self, mass, density) -> Tuple[float, float]:
        """Authored field coefficients ``(a, b)``, or raise, never guessed."""
        raise NotImplementedError(
            f"{type(self).__name__} ships no authored field coefficients; "
            "pass a, b manually or implement scalar_coeffs(mass, density)."
        )

    def to_embedding(self, coords):
        """Presentation transform (default: identity)."""
        return coords

    def boundary(self):
        """Presentation boundary paths (default: none)."""
        return []

    def display_axes(self):
        """Optional axis labels for the embedding (default: none authored).

        Return a list of strings, one per embedding column, so the viz layer
        does not guess geometry. Returning `None` (the default) means "no authored
        labels"; the viz layer falls back to generic `c0, c1, …`.
        """
        return None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        sig = cls.__dict__.get("signature")
        if isinstance(sig, tuple):
            if len(sig) != 2:
                raise TypeError(
                    f"{cls.__name__}: signature must be a (t, s) pair, got {sig!r}"
                )
            t, s = sig
            if t < 0 or s < 0:
                raise ValueError(
                    f"{cls.__name__}: signature entries must be non-negative, got {sig!r}"
                )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, type] = {}


def get_registry() -> dict:
    """A shallow copy of the spacetime name registry."""
    return dict(_registry)


def register(name: str, *, overwrite: bool = False):
    """Register a `Spacetime` subclass under ``name`` (its persistable identity).

    Duplicate names raise unless ``overwrite=True`` (no silent last-wins).
    """
    if not isinstance(name, str) or not name:
        raise ValueError("register() requires a non-empty name string.")

    def decorate(cls):
        if name in _registry and not overwrite:
            existing = _registry[name].__name__
            raise ValueError(
                f"Spacetime name {name!r} is already registered to "
                f"{existing}; pass overwrite=True to replace it."
            )
        _registry[name] = cls
        return cls

    return decorate


# ---------------------------------------------------------------------------
# Built-in Minkowski family (pure-Python, time-ordered by the sprinkler)
# ---------------------------------------------------------------------------


class MinkowskiDiamond(Spacetime):
    """Causal diamond in lightcone coordinates.

    For ``d == 2`` this is the true 1+1 causal diamond ``(u, v) \u2208 [0,1]\u00b2``.
    For ``d > 2`` the region is the product-interval ``[0,1]^d`` (a placeholder -
    the true ``I\u207a(p)\u2229I\u207b(q)`` sampler is tracked under R2_MINK).
    """

    def __init__(self, dimension: int = 2):
        self._dim = int(dimension)
        if self._dim < 2:
            raise ValueError("dimension must be >= 2")

    def dimension(self) -> int:
        return self._dim

    def volume(self) -> float:
        return 1.0

    def sample(self, rng, n):
        return rng.uniform(0.0, 1.0, size=(n, self._dim))

    def is_causal(self, u, v) -> bool:
        return bool(np.all(np.asarray(u) < np.asarray(v)))

    def is_causal_batch(self, coords):
        coords = np.asarray(coords, dtype=float)
        return np.all(coords[:, None, :] < coords[None, :, :], axis=2)

    def to_embedding(self, coords):
        coords = np.asarray(coords, dtype=float)
        if self._dim == 2:
            u = coords[:, 0]
            v = coords[:, 1]
            return np.column_stack(((u + v) / np.sqrt(2), (v - u) / np.sqrt(2)))
        return coords

    def boundary(self):
        if self._dim == 2:
            corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
            return [self.to_embedding(corners)]
        return []

    def display_axes(self):
        if self._dim == 2:
            return ["t", "x"]
        return None

    # back-compat aliases used by the visualization layer
    def transform_coordinates(self, coords):
        return self.to_embedding(coords)

    def get_boundary(self):
        return self.boundary()


class MinkowskiCylinder(Spacetime):
    """Flat Minkowski cylinder ``S\u00b9 \u00d7 \u211d`` (periodic spatial dimension)."""

    def __init__(self, dimension: int = 2, height: float = 1.0, circumference: float = 1.0):
        self._dim = int(dimension)
        if self._dim != 2:
            raise NotImplementedError("MinkowskiCylinder currently supports dimension 2 only")
        self.height = float(height)
        self.circumference = float(circumference)

    def dimension(self) -> int:
        return self._dim

    def volume(self) -> float:
        return self.height * self.circumference

    def sample(self, rng, n):
        t = rng.uniform(0.0, self.height, size=n)
        x = rng.uniform(0.0, self.circumference, size=n)
        return np.column_stack((t, x))

    def is_causal(self, u, v) -> bool:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        dt = v[0] - u[0]
        if dt <= 0.0:
            return False
        dx = abs(v[1] - u[1])
        dx = min(dx, self.circumference - dx)
        return bool(dt > dx)

    def is_causal_batch(self, coords):
        coords = np.asarray(coords, dtype=float)
        dt = coords[None, :, 0] - coords[:, None, 0]
        dx = np.abs(coords[None, :, 1] - coords[:, None, 1])
        dx = np.minimum(dx, self.circumference - dx)
        return (dt > 0) & (dt > dx)

    def to_embedding(self, coords):
        coords = np.asarray(coords, dtype=float)
        C = self.circumference
        R = C / (2 * np.pi)
        theta = (coords[:, 1] / C) * 2 * np.pi
        return np.column_stack((coords[:, 0], R * np.cos(theta), R * np.sin(theta)))

    def boundary(self):
        C = self.circumference
        H = self.height
        theta = np.linspace(0, 2 * np.pi, 100)
        x_flat = (theta / (2 * np.pi)) * C
        bottom = np.column_stack((np.zeros_like(x_flat), x_flat))
        top = np.column_stack((np.full_like(x_flat, H), x_flat))
        return [self.to_embedding(bottom), self.to_embedding(top)]

    def display_axes(self):
        return ["t", "x", "y"]  # 3D cylindrical embedding

    def transform_coordinates(self, coords):
        return self.to_embedding(coords)

    def get_boundary(self):
        return self.boundary()


class MinkowskiBox(Spacetime):
    """Rectangular block in flat Minkowski space (hard walls)."""

    def __init__(self, dimension: int = 2, time_extent: float = 1.0, space_extent: float = 1.0):
        self._dim = int(dimension)
        if self._dim < 2:
            raise ValueError("dimension must be >= 2")
        self.time_extent = float(time_extent)
        self.space_extent = float(space_extent)

    def dimension(self) -> int:
        return self._dim

    def volume(self) -> float:
        return self.time_extent * (self.space_extent ** (self._dim - 1))

    def sample(self, rng, n):
        pts = rng.uniform(0.0, 1.0, size=(n, self._dim))
        pts[:, 0] *= self.time_extent
        pts[:, 1:] *= self.space_extent
        return pts

    def is_causal(self, u, v) -> bool:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        dt = v[0] - u[0]
        if dt <= 0.0:
            return False
        dx_sq = float(np.sum((v[1:] - u[1:]) ** 2))
        return bool(dt * dt > dx_sq)

    def is_causal_batch(self, coords):
        coords = np.asarray(coords, dtype=float)
        dt = coords[None, :, 0] - coords[:, None, 0]
        dx_sq = np.sum((coords[None, :, 1:] - coords[:, None, 1:]) ** 2, axis=2)
        return (dt > 0) & (dt * dt > dx_sq)

    def to_embedding(self, coords):
        return np.asarray(coords, dtype=float)

    def boundary(self):
        if self._dim == 2:
            T = self.time_extent
            L = self.space_extent
            return [np.array([[0.0, 0.0], [T, 0.0], [T, L], [0.0, L], [0.0, 0.0]])]
        return []

    def display_axes(self):
        if self._dim == 2:
            return ["t", "x"]
        return None

    def transform_coordinates(self, coords):
        return self.to_embedding(coords)

    def get_boundary(self):
        return self.boundary()


register("minkowski_diamond")(MinkowskiDiamond)
register("minkowski_cylinder")(MinkowskiCylinder)
register("minkowski_box")(MinkowskiBox)


def _flat_scalar_coeffs(self, mass, density):
    """2D/4D flat-Minkowski scalar coefficients (authored data, never guessed)."""
    d = self.dimension()
    m = float(mass)
    if d == 2:
        return 0.5, -(m ** 2) / density
    if d == 4:
        return np.sqrt(density) / (2 * np.pi * np.sqrt(6)), -(m ** 2) / density
    raise NotImplementedError(
        f"scalar_coeffs for {d}D Minkowski spacetime are not implemented; "
        "pass a, b manually."
    )


MinkowskiDiamond.scalar_coeffs = _flat_scalar_coeffs
MinkowskiCylinder.scalar_coeffs = _flat_scalar_coeffs
MinkowskiBox.scalar_coeffs = _flat_scalar_coeffs


# ---------------------------------------------------------------------------
# Rung 0, declarative builder (flat Minkowski family)
# ---------------------------------------------------------------------------

_FLAT_DOMAINS = ("box", "diamond", "cylinder")
_SUPPORTED_METRICS = ("flat",)


def create(
    name: Optional[str] = None,
    *,
    dimension: int,
    signature: Optional[Tuple[int, int]] = None,
    domain: str,
    metric: str = "flat",
    **params,
):
    """Assemble a configured spacetime from a recipe (no class required).

    Every parameter maps 1:1 to a concrete setting, there is no hidden
    inference. Curved metrics (de_sitter / anti_de_sitter / flrw) are R2.1 and
    raise `NotImplementedError` here with the valid options.
    """
    if not isinstance(dimension, int) or dimension < 2:
        raise ValueError(f"dimension must be an int >= 2, got {dimension!r}")

    sig = (1, dimension - 1) if signature is None else tuple(signature)
    if len(sig) != 2 or (sig[0] + sig[1]) != dimension:
        raise ValueError(
            f"signature {sig!r} must satisfy t + s == dimension ({dimension})"
        )
    if sig[0] < 0 or sig[1] < 0:
        raise ValueError(f"signature entries must be non-negative, got {sig!r}")

    if metric != "flat":
        raise NotImplementedError(
            f"metric {metric!r} is not implemented yet (R2.1/R2_CURVED); "
            f"valid options: {list(_SUPPORTED_METRICS)}"
        )
    if domain not in _FLAT_DOMAINS:
        raise NotImplementedError(
            f"domain {domain!r} is not a supported flat domain; "
            f"valid options: {list(_FLAT_DOMAINS)}"
        )
    if sig[0] != 1:
        raise NotImplementedError(
            f"flat metric with signature {sig!r} is not implemented; only "
            "Lorentzian (1, d-1) flat spacetimes are available via create()."
        )

    if domain == "diamond":
        required = ()
    elif domain == "cylinder":
        required = ("height", "circumference")
    elif domain == "box":
        required = ("time_extent", "space_extent")
    else:  # pragma: no cover - guarded above
        required = ()

    missing = [p for p in required if params.get(p) is None]
    if missing:
        raise ValueError(f"domain={domain!r} requires parameters: {missing}")

    if domain == "diamond":
        return MinkowskiDiamond(dimension)
    if domain == "cylinder":
        return MinkowskiCylinder(dimension, params["height"], params["circumference"])
    if domain == "box":
        return MinkowskiBox(dimension, params["time_extent"], params["space_extent"])

    raise NotImplementedError(domain)  # pragma: no cover


# ---------------------------------------------------------------------------
# R2_CREATE, composition decorators + code generation
# ---------------------------------------------------------------------------


class TransformedSpacetime(Spacetime):
    """Wrap a spacetime with a coordinate transform (R2_CREATE decorator).

    ``forward`` maps base coordinates to the wrapped chart; ``inverse`` maps back.
    The transform is assumed **volume-preserving** (e.g. a translation or rotation);
    for non-volume-preserving maps, subclass and override ``volume()``.
    """

    def __init__(self, base: Spacetime, forward, inverse=None):
        self._base = base
        self._forward = forward
        self._inverse = inverse

    @property
    def signature(self) -> Tuple[int, int]:
        return self._base.signature

    def dimension(self) -> int:
        return self._base.dimension()

    def volume(self) -> float:
        return self._base.volume()

    def sample(self, rng, n):
        return self._forward(np.asarray(self._base.sample(rng, n), dtype=float))

    def is_causal(self, u, v) -> bool:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if self._inverse is not None:
            u = self._inverse(u)
            v = self._inverse(v)
        return bool(self._base.is_causal(u, v))

    def to_embedding(self, coords):
        return self._base.to_embedding(np.asarray(coords, dtype=float))

    def boundary(self):
        return self._base.boundary()

    def display_axes(self):
        return self._base.display_axes()


class RestrictedSpacetime(Spacetime):
    """Wrap a spacetime with a subregion predicate (R2_CREATE decorator).

    ``region(coords) -> bool`` keeps a point. Sampling uses rejection; ``is_causal``
    is inherited. ``volume`` is either provided explicitly or estimated by Monte
    Carlo so that ``volume ↔ sample`` stay consistent.
    """

    def __init__(self, base: Spacetime, region, volume=None):
        self._base = base
        self._region = region
        self._volume = float(volume) if volume is not None else self._estimate_volume()

    @property
    def signature(self) -> Tuple[int, int]:
        return self._base.signature

    def _estimate_volume(self, n: int = 20000) -> float:
        rng = np.random.default_rng(0)
        pts = self._base.sample(rng, n)
        accepted = sum(1 for p in pts if bool(self._region(p)))
        return self._base.volume() * (accepted / n)

    def dimension(self) -> int:
        return self._base.dimension()

    def volume(self) -> float:
        return self._volume

    def sample(self, rng, n):
        out = []
        while len(out) < n:
            for p in self._base.sample(rng, n):
                if bool(self._region(p)):
                    out.append(p)
                    if len(out) == n:
                        break
        return np.asarray(out, dtype=float)

    def is_causal(self, u, v) -> bool:
        return self._base.is_causal(u, v)

    def to_embedding(self, coords):
        return self._base.to_embedding(coords)

    def boundary(self):
        return self._base.boundary()

    def display_axes(self):
        return self._base.display_axes()


class ConformalSpacetime(Spacetime):
    """Wrap a spacetime with a conformal factor (R2_CREATE decorator).

    A conformal transformation rescales the metric by ``Omega(x)^2``. It preserves
    the **causal light-cone**, so ``is_causal`` is inherited verbatim, but rescales
    the volume measure by ``Omega^d``.

    ``conformal_factor(x) -> float`` must be positive on the base's support (points
    where it returns ``<= 0`` are rejected). ``volume`` is given explicitly or
    Monte-Carlo estimated as ``E[Omega^d] * V_base``; ``sample`` rejection-samples
    with weight ``Omega^d`` so ``volume ↔ sample`` stay consistent. Because the
    rejection bound is calibrated by sampling, pass ``max_weight >= sup(Omega^d)``
    explicitly when the factor has sharp, poorly-sampled peaks.
    """

    def __init__(self, base: Spacetime, conformal_factor, volume=None, max_weight=None):
        self._base = base
        self._omega = conformal_factor
        self._d = base.dimension()
        self._max_weight = max_weight
        self._volume = float(volume) if volume is not None else self._estimate_volume()

    @property
    def signature(self) -> Tuple[int, int]:
        return self._base.signature

    def _estimate_volume(self, n: int = 20000) -> float:
        rng = np.random.default_rng(0)
        pts = np.asarray(self._base.sample(rng, n), dtype=float)
        om = np.asarray([self._omega(p) for p in pts], dtype=float)
        weights = np.power(np.maximum(om, 0.0), self._d)
        return self._base.volume() * float(np.mean(weights))

    def _calibrate_max_weight(self, n: int = 20000) -> float:
        rng = np.random.default_rng(1)
        pts = np.asarray(self._base.sample(rng, n), dtype=float)
        om = np.asarray([self._omega(p) for p in pts], dtype=float)
        if np.all(om <= 0.0):
            raise ValueError("conformal factor is non-positive everywhere; cannot sample")
        return float(np.max(om) ** self._d)

    def dimension(self) -> int:
        return self._d

    def volume(self) -> float:
        return self._volume

    def is_causal(self, u, v) -> bool:
        return bool(self._base.is_causal(u, v))

    def sample(self, rng, n):
        max_w = self._max_weight if self._max_weight is not None else self._calibrate_max_weight()
        if max_w <= 0.0:
            raise ValueError("conformal max_weight must be > 0")
        out = []
        while len(out) < n:
            pts = np.asarray(self._base.sample(rng, n), dtype=float)
            for p in pts:
                om = float(self._omega(p))
                if om <= 0.0:
                    continue
                w = om ** self._d
                if rng.random() < min(1.0, w / max_w):
                    out.append(p)
                    if len(out) == n:
                        break
        return np.asarray(out, dtype=float)

    def to_embedding(self, coords):
        return self._base.to_embedding(coords)

    def boundary(self):
        return self._base.boundary()

    def display_axes(self):
        return self._base.display_axes()


class PeriodicSpacetime(Spacetime):
    """Wrap a spacetime with periodic identification along spacelike axes (R2_CREATE).

    ``periods`` maps an axis index to its period ``L > 0``; points are identified
    ``x[a] ~ x[a] + L`` and live in the fundamental domain ``[0, L)`` along that
    axis. A bare number is shorthand for "wrap every spacelike axis with that
    period". ``sample`` wraps base samples into the fundamental domain; ``volume``
    is the base volume (the fundamental domain).

    The quotient causal order is ``u ≺ v`` iff *some* periodic image of ``v`` lies
    in the future of ``u`` (equivalently some image of ``u`` in the past of ``v``),
    checked within ``±max_images`` shifts per periodic axis. Only **spacelike**
    axes may be periodic: periodic time would produce closed timelike curves, so
    axis ``0`` raises rather than silently shipping a pathological order.
    """

    def __init__(self, base: Spacetime, periods, max_images: int = 3):
        self._base = base
        d = base.dimension()
        if isinstance(periods, (int, float)):
            periods = {a: float(periods) for a in range(1, d)}
        else:
            periods = dict(periods)

        cleaned = {}
        for axis, L in periods.items():
            axis = int(axis)
            L = float(L)
            if axis == 0:
                raise NotImplementedError(
                    "periodic time would create closed timelike curves; only "
                    "spacelike axes (index >= 1) may be periodic"
                )
            if axis < 0 or axis >= d:
                raise ValueError(f"periodic axis {axis} out of range for dimension {d}")
            if L <= 0.0:
                raise ValueError(f"period must be > 0, got {L!r}")
            cleaned[axis] = L
        self._periods = cleaned
        self._max_images = int(max_images)

    @property
    def signature(self) -> Tuple[int, int]:
        return self._base.signature

    def dimension(self) -> int:
        return self._base.dimension()

    def volume(self) -> float:
        return self._base.volume()

    def sample(self, rng, n):
        pts = np.asarray(self._base.sample(rng, n), dtype=float)
        for axis, L in self._periods.items():
            pts[:, axis] = np.mod(pts[:, axis], L)
        return pts

    def is_causal(self, u, v) -> bool:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        axes = sorted(self._periods)
        shift_ranges = [range(-self._max_images, self._max_images + 1) for _ in axes]
        for combo in itertools.product(*shift_ranges):
            v_img = v.copy()
            for axis, k in zip(axes, combo):
                v_img[axis] += k * self._periods[axis]
            if bool(self._base.is_causal(u, v_img)):
                return True
        return False

    def to_embedding(self, coords):
        return self._base.to_embedding(coords)

    def boundary(self):
        return self._base.boundary()

    def display_axes(self):
        return self._base.display_axes()


def _recipe_from_spacetime(st) -> dict:
    if isinstance(st, MinkowskiDiamond):
        return {"name": "MinkowskiDiamond", "dimension": st.dimension(), "domain": "diamond"}
    if isinstance(st, MinkowskiCylinder):
        return {
            "name": "MinkowskiCylinder",
            "dimension": st.dimension(),
            "domain": "cylinder",
            "height": st.height,
            "circumference": st.circumference,
        }
    if isinstance(st, MinkowskiBox):
        return {
            "name": "MinkowskiBox",
            "dimension": st.dimension(),
            "domain": "box",
            "time_extent": st.time_extent,
            "space_extent": st.space_extent,
        }
    raise NotImplementedError(f"export_python for {type(st).__name__} is not implemented")


def export_python(recipe_or_st) -> str:
    """Emit a paste-ready `Spacetime` subclass for a recipe (R2_CREATE codegen).

    The emitted subclass delegates to `create(recipe)`, the same template `create`
    uses, so it can never drift from the declarative builder.
    """
    if isinstance(recipe_or_st, Spacetime):
        recipe = _recipe_from_spacetime(recipe_or_st)
    elif isinstance(recipe_or_st, dict):
        recipe = dict(recipe_or_st)
    else:
        raise TypeError("export_python expects a recipe dict or a Spacetime instance")

    name = recipe.get("name", "MySpacetime")
    dimension = recipe.get("dimension")
    if dimension is None:
        raise ValueError("export_python: recipe requires 'dimension'")
    domain = recipe.get("domain", "diamond")
    metric = recipe.get("metric", "flat")
    signature = recipe.get("signature")

    reserved = {"name", "dimension", "domain", "metric", "signature"}
    params = {k: v for k, v in recipe.items() if k not in reserved}

    call_args = [f"dimension={dimension!r}", f"domain={domain!r}", f"metric={metric!r}"]
    if signature is not None:
        call_args.append(f"signature={signature!r}")
    for k, v in params.items():
        call_args.append(f"{k}={v!r}")

    lines = [
        f"@spacetime.register({name!r})",
        f"class {name}(spacetime.Spacetime):",
        f'    """Generated by spacetime.export_python({name!r})."""',
        "",
        "    def __init__(self):",
        f"        self._impl = spacetime.create({', '.join(call_args)})",
        "",
        "    def dimension(self): return self._impl.dimension()",
        "    def volume(self): return self._impl.volume()",
        "    def sample(self, rng, n): return self._impl.sample(rng, n)",
        "    def is_causal(self, u, v): return self._impl.is_causal(u, v)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# R2_CURVED, curved / cosmological spacetimes (documented parametrizations)
# ---------------------------------------------------------------------------


def _unit_sphere(angles):
    """Map ``(n, m)`` spherical angles to ``(n, m+1)`` unit vectors on ``S^m``."""
    angles = np.asarray(angles, dtype=float)
    if angles.ndim == 1:
        angles = angles[None, :]
    n, m = angles.shape
    out = np.empty((n, m + 1))
    sin_prod = np.ones(n)
    for k in range(m):
        out[:, k] = sin_prod * np.cos(angles[:, k])
        sin_prod = sin_prod * np.sin(angles[:, k])
    out[:, m] = sin_prod
    return out


def _trapz(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


class DeSitter(Spacetime):
    """de Sitter spacetime: the hyperboloid ``-X\u2080\u00b2 + \u03a3 X\u1d62\u00b2 = R\u00b2`` in Minkowski_{d+1}.

    Global coordinates ``(t, \u03a9)``; ``is_causal`` is the ambient-Minkowski order.
    The sampler is a documented **parametrization** (not the dS-invariant measure);
    ``scalar_coeffs`` raises (manual ``a, b``).
    """

    def __init__(self, dimension: int = 2, radius: float = 1.0, time_extent: float = 2.0):
        self._d = int(dimension)
        if self._d < 2:
            raise ValueError("dimension must be >= 2")
        self.R = float(radius)
        self.T = float(time_extent)

    def dimension(self) -> int:
        return self._d

    def volume(self) -> float:
        # Finite patch t in [-T, T]: V = R^d * A_{d-1} * ∫ cosh^{d-1}(t) dt.
        t = np.linspace(-self.T, self.T, 2000)
        A = 2 * math.pi ** (self._d / 2) / math.gamma(self._d / 2)  # area of S^{d-1}
        return self.R ** self._d * A * _trapz(np.cosh(t) ** (self._d - 1), t)

    def sample(self, rng, n):
        t = rng.uniform(-self.T, self.T, size=n)
        if self._d == 2:
            angles = rng.uniform(0, 2 * np.pi, size=(n, 1))
        else:
            angles = np.empty((n, self._d - 1))
            angles[:, :-1] = rng.uniform(0, np.pi, size=(n, self._d - 2))
            angles[:, -1] = rng.uniform(0, 2 * np.pi, size=n)
        return np.column_stack([t, angles])

    def is_causal(self, u, v) -> bool:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        t1, t2 = u[0], v[0]
        if t2 <= t1:
            return False
        w1 = _unit_sphere(u[1:])[0]
        w2 = _unit_sphere(v[1:])[0]
        cosg = float(w1 @ w2)
        return bool(np.cosh(t1) * np.cosh(t2) * cosg - np.sinh(t1) * np.sinh(t2) >= 1.0)


class AntiDeSitter(Spacetime):
    """anti-de Sitter spacetime: hyperboloid ``-X\u2080\u00b2 - X\u2081\u00b2 + \u03a3 X\u1d62\u00b2 = -R\u00b2``.

    The **naive** hyperboloid has closed timelike curves, so ``is_causal`` raises
    ("no causal order"); the universal cover is a research task (R2_CURVED caveat).
    The sampler is a documented parametrization; ``scalar_coeffs`` raises.
    """

    def __init__(self, dimension: int = 2, radius: float = 1.0, rho_max: float = 1.0):
        self._d = int(dimension)
        if self._d < 2:
            raise ValueError("dimension must be >= 2")
        self.R = float(radius)
        self.rho_max = float(rho_max)

    def dimension(self) -> int:
        return self._d

    def volume(self) -> float:
        raise NotImplementedError("AdS volume for the finite patch is not implemented")

    def sample(self, rng, n):
        t = rng.uniform(0, 2 * np.pi, size=n)
        rho = rng.uniform(0, self.rho_max, size=n)
        if self._d == 2:
            return np.column_stack([t, rho])
        angles = rng.uniform(0, 2 * np.pi, size=(n, self._d - 2))
        return np.column_stack([t, rho, angles])

    def is_causal(self, u, v) -> bool:
        raise NotImplementedError(
            "the naive AntiDeSitter hyperboloid has closed timelike curves; "
            "a causal order needs the universal cover (not implemented)."
        )


class FLRW(Spacetime):
    """FLRW spacetime (flat spatial slices, k=0): ``ds\u00b2 = -dt\u00b2 + a(t)\u00b2 dx\u20d7\u00b2``.

    ``scale_factor`` is a power-law exponent ``p`` (``a(t) = t^p``) or a callable
    ``a(t) -> float``. Causality uses the null condition ``\u222b dt/a(t) \u2265 |\u0394x\u20d7|``.
    The sampler is uniform in ``(t, x\u20d7)``, a documented parametrization, not the
    FLRW-invariant measure unless ``a(t)`` is constant.
    """

    def __init__(self, dimension: int = 2, scale_factor=0, time_extent: float = 1.0,
                 space_extent: float = 1.0):
        self._d = int(dimension)
        if self._d < 2:
            raise ValueError("dimension must be >= 2")
        self._p = None if callable(scale_factor) else float(scale_factor)
        self._a = scale_factor if callable(scale_factor) else None
        self.T = float(time_extent)
        self.L = float(space_extent)
        self._t0 = 1e-6 if (self._p is not None and self._p >= 1.0) else 0.0

    def dimension(self) -> int:
        return self._d

    def _scale(self, t):
        return self._a(t) if self._a is not None else t ** self._p

    def _horizon(self, t1, t2):
        """Comoving horizon ``\u222b_{t1}^{t2} dt / a(t)``."""
        if self._a is not None:
            ts = np.linspace(t1, t2, 200)
            return _trapz(1.0 / np.array([self._scale(x) for x in ts]), ts)
        p = self._p
        if p == 0:
            return t2 - t1
        if p == 1:
            return math.log(t2 / t1)
        return (t2 ** (1 - p) - t1 ** (1 - p)) / (1 - p)

    def volume(self) -> float:
        ts = np.linspace(self._t0, self.T, 2000)
        integrand = np.array([self._scale(x) for x in ts]) ** (self._d - 1)
        return self.L ** (self._d - 1) * _trapz(integrand, ts)

    def sample(self, rng, n):
        t = rng.uniform(self._t0, self.T, size=n)
        x = rng.uniform(0.0, self.L, size=(n, self._d - 1))
        return np.column_stack([t, x])

    def is_causal(self, u, v) -> bool:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        t1, t2 = u[0], v[0]
        if t2 <= t1:
            return False
        dx = float(np.linalg.norm(v[1:] - u[1:]))
        return self._horizon(t1, t2) >= dx


register("de_sitter")(DeSitter)
register("anti_de_sitter")(AntiDeSitter)
register("flrw")(FLRW)


def _tortoise(r, M):
    r = np.asarray(r, dtype=float)
    return r + 2.0 * M * np.log(r / (2.0 * M) - 1.0)


class Schwarzschild(Spacetime):
    """Schwarzschild black hole (R2_BH), geometry-only, 1+1 (radial) exact.

    Exterior region ``r > 2M`` in Schwarzschild coordinates ``(t, r)``. ``is_causal``
    uses the **exact** radial null condition via the tortoise coordinate
    ``r* = r + 2M ln(r/2M - 1)``. Higher dimensions (the angular null geodesic) are a
    research task and raise. ``scalar_coeffs`` raises (manual ``a, b``).
    """

    def __init__(self, dimension: int = 2, mass: float = 1.0, r_max: float = 10.0,
                 time_extent: float = 10.0):
        self._d = int(dimension)
        if self._d != 2:
            raise NotImplementedError(
                "Schwarzschild currently supports 1+1 (dimension=2); the angular null "
                "geodesic for d > 2 is a research task."
            )
        self.M = float(mass)
        self.r_min = 2.0 * self.M * 1.001
        self.r_max = float(r_max)
        self.T = float(time_extent)

    def dimension(self) -> int:
        return self._d

    def volume(self) -> float:
        raise NotImplementedError("Schwarzschild volume for the exterior patch is not implemented")

    def sample(self, rng, n):
        t = rng.uniform(0.0, self.T, size=n)
        r = rng.uniform(self.r_min, self.r_max, size=n)
        return np.column_stack([t, r])

    def is_causal(self, u, v) -> bool:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        t1, r1 = u[0], u[1]
        t2, r2 = v[0], v[1]
        if t2 <= t1:
            return False
        dtau = t2 - t1
        drstar = abs(float(_tortoise(r2, self.M) - _tortoise(r1, self.M)))
        return bool(dtau >= drstar)


register("schwarzschild")(Schwarzschild)
