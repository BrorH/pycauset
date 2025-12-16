import abc
import math
from typing import Tuple, Optional
from .causet import CausalSet
# from . import compute_k # Circular import

class Field(abc.ABC):
    """
    Abstract base class for fields defined on a Causal Set.
    
    A Field represents the matter content (or vacuum state) imposed on the 
    spacetime geometry of a Causal Set. It is responsible for defining 
    the propagators and correlation functions associated with that field.
    """
    def __init__(self, causet: CausalSet):
        self._causet = causet

    @property
    def causet(self) -> CausalSet:
        return self._causet

    @abc.abstractmethod
    def propagator(self):
        """Compute the propagator (Green's function) for this field."""
        pass


class ScalarField(Field):
    """
    A massive scalar field defined on a Causal Set.
    
    The retarded propagator K_R is defined as:
        K_R = Phi * (I - b * Phi)^-1
    where Phi = a * C.
    
    The coefficients 'a' and 'b' are derived from the spacetime dimension,
    sprinkling density, and field mass.
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
        """
        Initialize a scalar field.
        
        Args:
            causet: The causal set on which the field lives.
            mass: The mass of the field. Defaults to 0.0 (massless).
            n: Convenience constructor: if provided (and `causet` is None), a new CausalSet
               is sprinkled with this many elements.
            density: Convenience constructor: if provided (and `causet` is None), a new CausalSet
               is sprinkled with Poisson(density * volume) elements.
            spacetime: Optional spacetime for the sprinkled CausalSet.
            seed: Optional RNG seed for sprinkling.
            matrix: Optional pre-existing causal matrix to attach to the CausalSet.
        """
        if causet is None:
            causet = CausalSet(n=n, density=density, spacetime=spacetime, seed=seed, matrix=matrix)

        super().__init__(causet)
        self._mass = float(mass)
        self._cached_propagator = None
        
        # Validate that we have the necessary info to compute physics
        if self._causet.density is None:
            # If density is missing (e.g. raw matrix load), we can't compute a/b automatically.
            # We don't raise here, but propagator() will fail unless a/b are manually provided.
            pass

    @property
    def mass(self) -> float:
        return self._mass

    def _get_coeffs(self) -> Tuple[float, float]:
        """
        Calculate the coefficients 'a' and 'b' based on spacetime and mass.
        
        Returns:
            (a, b)
        """
        # We need density to compute coefficients
        try:
            rho = self._causet.density
        except (ValueError, AttributeError):
            rho = None
            
        if rho is None:
             raise ValueError(
                "Cannot compute field coefficients: CausalSet density is unknown. "
                "Ensure the CausalSet was created with density information or provide 'a' and 'b' manually."
            )

        dim = self._causet.spacetime.dimension()
        st_type = self._causet.spacetime.__class__.__name__
        
        # Check if spacetime is Minkowski-like (flat)
        # This includes Diamond, Cylinder, Box, etc.
        is_flat = "Minkowski" in st_type
        
        if not is_flat:
             raise NotImplementedError(
                f"Field coefficients for spacetime '{st_type}' are not yet implemented. "
                "Please provide 'a' and 'b' manually."
            )

        m = self._mass
        
        if dim == 2:
            # 1+1 Dimensions
            a = 0.5
            b = - (m**2) / rho
        elif dim == 4:
            # 3+1 Dimensions
            # a = sqrt(rho) / (2 * pi * sqrt(6))
            a = math.sqrt(rho) / (2 * math.pi * math.sqrt(6))
            b = - (m**2) / rho
        else:
            raise NotImplementedError(
                f"Field coefficients for {dim}D Minkowski spacetime are not yet implemented. "
                "Please provide 'a' and 'b' manually."
            )
            
        return a, b

    def compute_retarded_propagator(self, a: Optional[float] = None, b: Optional[float] = None):
        """
        Compute the Retarded Propagator K_R.
        
        Args:
            a (float, optional): Manual override for coefficient 'a'.
            b (float, optional): Manual override for coefficient 'b'.
            
        Returns:
            TriangularFloatMatrix: The K_R matrix.
        """
        if self._cached_propagator is not None and a is None and b is None:
            return self._cached_propagator

        # 1. Determine coefficients
        if a is None or b is None:
            calc_a, calc_b = self._get_coeffs()
            if a is None: a = calc_a
            if b is None: b = calc_b
            
        C = self._causet.C
        
        # 2. Handle Massless Limit (b -> 0)
        # If mass is 0, b is 0.
        if abs(b) < 1e-15:
            # K_R = a * C
            result = a * C
        else:
            # 3. Compute Massive Propagator
            # Formula: K_R = (-1/b) * C * (alpha_eff * I + C)^-1
            # Where alpha_eff = -1 / (a * b)
            
            alpha_eff = -1.0 / (a * b)
            
            # X = C * (alpha_eff * I + C)^-1
            # This is exactly what compute_k(C, alpha) calculates
            from . import compute_k
            X = compute_k(C, alpha_eff)
            
            # Result is (-1/b) * X
            result = (-1.0 / b) * X
        
        if a is None and b is None: # Only cache if using default coefficients
            self._cached_propagator = result
            
        return result

    def propagator(self, a: Optional[float] = None, b: Optional[float] = None):
        """Alias for compute_retarded_propagator."""
        return self.compute_retarded_propagator(a, b)

    def pauli_jordan(self):
        """
        Compute the Pauli-Jordan function i*Delta, where Delta = K - K^T.
        
        Returns:
            AntiSymmetricFloat64Matrix: The matrix Delta, with scalar set to 1j (imaginary unit).
            This represents the operator i*Delta.
        """
        K = self.compute_retarded_propagator()
        
        # Create Delta = K - K^T efficiently using AntiSymmetricMatrix.from_triangular
        # This copies the upper triangle of K into an AntiSymmetricMatrix.
        # Since K is triangular, K - K^T is exactly what AntiSymmetricMatrix represents
        # when initialized from K (assuming K is upper triangular).
        # If K is lower triangular, we might need to transpose first, but standard K is upper packed?
        # Wait, TriangularMatrix usually stores upper triangle.
        # If K is Retarded, it is non-zero for causal future.
        # If indices are sorted by time, future is i < j (upper) or i > j (lower)?
        # In PyCauset, C[i, j] = 1 if i < j and i prec j. So C is Upper Triangular.
        # So K is Upper Triangular.
        # So Delta = K - K^T has K in upper triangle and -K^T in lower.
        # AntiSymmetricMatrix stores upper triangle and negates for lower access.
        # So AntiSymmetricMatrix(K) effectively represents K - K^T.
        
        from . import AntiSymmetricFloat64Matrix
        Delta = AntiSymmetricFloat64Matrix.from_triangular(K)
        
        # Set scalar to i (imaginary unit) to represent i*Delta
        # We use a complex scalar if supported, or just document it.
        # The user requested storing the factor of i in the scalar.
        Delta.set_scalar(1j)
        
        return Delta
