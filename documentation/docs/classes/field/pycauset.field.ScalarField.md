# pycauset.field.ScalarField

```python
class ScalarField(Field)
```

Represents a massive scalar field defined on a Causal Set.

This class implements the generalized Retarded Propagator $K_R$ for a scalar field $\phi$ satisfying the Klein-Gordon equation on the discrete causal set structure.

## Constructor

```python
ScalarField(causet: CausalSet, mass: float = 0.0)
```

*   **causet** (*CausalSet*): The causal set on which the field lives. Must have density information available.
*   **mass** (*float*): The mass of the field ($m$). Defaults to 0.0 (massless).

## Methods

### propagator

```python
def propagator(self, a: float = None, b: float = None) -> TriangularFloatMatrix
```

Computes the Retarded Propagator $K_R$.

The propagator is defined as:
$$ K_R = \Phi(I - b\Phi)^{-1} $$
where $\Phi = a C$.

**Automatic Coefficient Derivation:**
If `a` and `b` are not provided, they are automatically calculated based on the `spacetime` dimension ($d$), the sprinkling density ($\rho$), and the field mass ($m$).

*   **For $d=2$ (Minkowski):**
    $$ a = 1/2, \quad b = -m^2/\rho $$
*   **For $d=4$ (Minkowski):**
    $$ a = \frac{\sqrt{\rho}}{2\pi\sqrt{6}}, \quad b = -m^2/\rho $$

**Parameters:**

*   **a** (*float, optional*): Manual override for coefficient $a$.
*   **b** (*float, optional*): Manual override for coefficient $b$.

**Returns:**

*   **TriangularFloatMatrix**: The computed propagator matrix.

**Raises:**

*   **ValueError**: If the causal set density is unknown and coefficients are not manually provided.
*   **NotImplementedError**: If the spacetime dimension/type is not supported for automatic derivation.

### pauli_jordan

```python
def pauli_jordan(self) -> AntiSymmetricFloat64Matrix
```

Computes the Pauli-Jordan function $i\Delta$, where $\Delta = K - K^T$.

This function returns an `AntiSymmetricFloat64Matrix` representing the operator $i\Delta$. The matrix stores the values of $\Delta$, but its `scalar` property is set to `1j` (the imaginary unit), so that accessing elements or performing arithmetic operations treats it as $i\Delta$.

**Returns:**

*   **AntiSymmetricFloat64Matrix**: The matrix $i\Delta$.
