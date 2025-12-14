# Field Theory on Causal Sets

Causal Set Theory is not just about the discrete structure of spacetime itself; it is also a framework for doing Quantum Field Theory (QFT). To do this, we define **Fields** that live on the causal set.

In PyCauset, the geometry (the [[pycauset.CausalSet]]) and the matter (the [[pycauset.field.Field]]) are distinct objects. This separation allows you to study different fields (massless, massive, interacting) on the same underlying spacetime background.

## The Scalar Field

The most fundamental field studied in Causal Set Theory is the **Scalar Field**. 
### The Retarded Propagator ($K_R$)

The Retarded Propagator $K_R$ is the inverse of the causal set d'Alembertian. In PyCauset, it is computed using the generalized formula:

$$ K_R = \Phi(I - b\Phi)^{-1} $$

where $\Phi = a C$ ($C$ is the Causal Matrix).

### Parameters

The coefficients $a$ and $b$ are the bridge between the discrete matrix mathematics and the continuous physical parameters. They depend on:

1.  **Dimension ($d$)**: The dimension of the manifold (e.g., 2 or 4).
2.  **Density ($\rho$)**: The sprinkling density ($N/V$).
3.  **Mass ($m$)**: The mass of the field.

PyCauset automatically derives $a$ and $b$ for standard Minkowski spacetimes.

| Dimension | $a$ | $b$ |
| :--- | :--- | :--- |
| **2D** | $1/2$ | $-m^2/\rho$ |
| **4D** | $\frac{\sqrt{\rho}}{2\pi\sqrt{6}}$ | $-m^2/\rho$ |

## Usage Guide

### 1. Define the Spacetime and Causal Set
First, generate your background geometry. Note that you **must** use density-based sprinkling (or provide the density manually) for the field coefficients to be calculated correctly.

```python
import pycauset as pc

# 1. Define Spacetime (2D Minkowski)
st = pc.spacetime.MinkowskiDiamond(2)

# 2. Sprinkle Causal Set (Density is required!)
c = pc.CausalSet(density=1000, spacetime=st)
```

### 2. Define the Field
Create a [[pycauset.field.ScalarField]] instance attached to your causal set. This is where you specify the mass.

```python
from pycauset.field import ScalarField

# Define a massive field (m=1.5)
field = ScalarField(c, mass=1.5)
```

### 3. Compute the Propagator
Call `.propagator()` to compute the $K_R$ matrix. This operation is computationally intensive ($O(N^2)$) and returns a [[pycauset.TriangularFloatMatrix]].

```python
# Compute K_R
K = field.propagator()

# K is a large matrix backed by disk storage
print(f"Propagator shape: {K.shape}")
```

### Massless Limit
For a massless field ($m=0$), the parameter $b$ becomes 0. The formula simplifies to $K_R = aC$.

```python
massless_field = ScalarField(c, mass=0.0)
K_0 = massless_field.propagator()
```

## Advanced: Manual Coefficients
If you are working with a non-standard spacetime or want to experiment with different non-locality parameters, you can override $a$ and $b$ manually.

```python
# Manually specifying coefficients (bypasses automatic derivation)
K_custom = field.propagator(a=0.5, b=-0.02)
```

## The Pauli-Jordan Function ($i\Delta$)

The Pauli-Jordan function $\Delta$ is defined as the difference between the retarded and advanced propagators:
$$ \Delta = K_R - K_A $$
Since $K_A = K_R^T$, this is equivalent to:
$$ \Delta = K - K^T $$

In Quantum Field Theory, the operator of interest is often $i\Delta$. PyCauset provides a dedicated method to compute this efficiently.

```python
# Compute i*Delta
Delta = field.pauli_jordan()
```

The result is an `AntiSymmetricFloat64Matrix`. To represent the factor of $i$ without storing complex numbers for every element (which would double the storage requirement), PyCauset stores the real values of $\Delta$ and sets the matrix's `scalar` property to `1j`.

When you access elements or perform arithmetic with this matrix, the factor of $i$ is automatically applied.
