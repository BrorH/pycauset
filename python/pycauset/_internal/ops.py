from __future__ import annotations

import functools

from typing import Any

from .coercion import safe_rows_cols as _safe_rows_cols


def _as_pycauset_array(obj: Any, *, deps: OpsDeps) -> Any:
    return deps.native._asarray(obj)


class OpsDeps:
    def __init__(
        self,
        *,
        native: Any,
        np_module: Any | None,
        Matrix: Any,
        warnings_module: Any,
        io_observer: Any | None,
    ) -> None:
        self.native = native
        self.np_module = np_module
        self.Matrix = Matrix
        self.warnings = warnings_module
        self.io_observer = io_observer


def _traced(op: str, n_operands: int = 1):
    """Write an IO trace around the call (F-100): the first `n_operands` positional
    arguments are the operands; the engine's kernel log says what ran."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args: Any, deps: OpsDeps, **kwargs: Any) -> Any:
            observer = deps.io_observer
            if observer is None:
                return fn(*args, deps=deps, **kwargs)
            with observer.trace(op, list(args[:n_operands])):
                return fn(*args, deps=deps, **kwargs)
        return inner
    return wrap


@_traced("matmul", 2)
def matmul(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    # Phase F integration: BlockMatrix routing.
    # If either operand is a BlockMatrix, preserve 'once block, always block'
    # by returning a thunked BlockMatrix via block orchestration.
    from .blockmatrix import BlockMatrix, block_matmul

    if isinstance(a, BlockMatrix) or isinstance(b, BlockMatrix):
        if not isinstance(a, BlockMatrix):
            a = BlockMatrix([[a]])
        if not isinstance(b, BlockMatrix):
            b = BlockMatrix([[b]])
        return block_matmul(a, b)

    native_matrix_base = deps.native.MatrixBase
    native_vector_base = deps.native.VectorBase

    # NumPy-like behavior: allow vectors in matmul by deferring to the native
    # operator implementation (which encodes the 1D rules).
    if native_vector_base is not None and (
        isinstance(a, native_vector_base) or isinstance(b, native_vector_base)
    ):
        return a @ b

    if native_matrix_base is not None:
        if isinstance(a, native_matrix_base) and isinstance(b, native_matrix_base):
            # The engine dispatches every structure and dtype itself. The conversions that
            # were here (symmetric -> dense float64, triangular/diagonal -> float64) changed
            # result dtypes and duplicated engine paths (F-073).
            return a @ b

    # Array-likes (NumPy arrays, nested lists): import, then the engine. No per-element or
    # NumPy fallback (R1: NumPy is an oracle, not a backend).
    if not isinstance(a, (native_matrix_base, native_vector_base)) or not isinstance(
        b, (native_matrix_base, native_vector_base)
    ):
        return matmul(_import_operand(a, deps=deps), _import_operand(b, deps=deps), deps=deps)
    raise TypeError(f"matmul: unsupported operands {type(a).__name__}, {type(b).__name__}")


def _import_operand(x: Any, *, deps: OpsDeps) -> Any:
    """The one array-like rule of the public functions (review 3.8 C-10): a PyCauset matrix or
    vector as is; a NumPy array or nested lists imported (1-D: a vector, 2-D: a matrix); a
    BlockMatrix refused (`_engine_matrix`'s rule: it is not densified behind the caller's back)."""
    native = (deps.native.MatrixBase, deps.native.VectorBase)
    if isinstance(x, native):
        return x
    if _is_block(x):
        return _engine_matrix(x, deps=deps)  # raises the explicit TypeError
    arr = deps.np_module.asarray(x)
    if arr.ndim == 1:
        return deps.native._asarray(arr)
    return deps.Matrix(arr)


def compute_k(matrix: Any, a: float, *, deps: OpsDeps) -> Any:
    return deps.native._compute_k_matrix(_engine_matrix(matrix, deps=deps), float(a))


def _is_block(x: Any) -> bool:
    from .blockmatrix import BlockMatrix

    return isinstance(x, BlockMatrix)


def _is_scalar(x: Any, deps: OpsDeps) -> bool:
    return isinstance(x, (bool, int, float, complex, deps.np_module.generic))


def elementwise(op: Any, a: Any, b: Any, name: str, *, deps: OpsDeps) -> Any:
    """`op(a, b)` in the engine for the function forms (`add`, `divide`, `bitwise_and`, ...):
    array-like operands (NumPy arrays, nested lists) are imported first, scalars stay scalars.
    `pc.divide(ndarray, ndarray)` computed in NumPy and returned an ndarray (review 3.8 C-4)."""
    if _is_scalar(a, deps) and _is_scalar(b, deps):
        raise TypeError(f"{name}: at least one operand must be a matrix, vector or array")
    if _is_block(a) or _is_block(b):
        return op(a, b)  # block operators ("once block, always block"; it densified, 3.10 C-7)
    a = a if _is_scalar(a, deps) else _import_operand(a, deps=deps)
    b = b if _is_scalar(b, deps) else _import_operand(b, deps=deps)
    return op(a, b)


def bitwise_not(matrix: Any, *, deps: OpsDeps) -> Any:
    """~x (NumPy invert): logical not for bool, bitwise not for integers. Engine only."""
    if _is_scalar(matrix, deps):
        raise TypeError("bitwise_not: the operand must be a matrix, vector or array")
    return ~_import_operand(matrix, deps=deps)


@_traced("bitwise_and", 2)
def bitwise_and(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """a & b with NumPy's dtype rules (bool & bool is bool; integers promote). Engine only."""
    return elementwise(lambda x, y: x & y, a, b, "bitwise_and", deps=deps)


@_traced("bitwise_or", 2)
def bitwise_or(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """a | b with NumPy's dtype rules. Engine only."""
    return elementwise(lambda x, y: x | y, a, b, "bitwise_or", deps=deps)


@_traced("bitwise_xor", 2)
def bitwise_xor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """a ^ b with NumPy's dtype rules. Engine only."""
    return elementwise(lambda x, y: x ^ y, a, b, "bitwise_xor", deps=deps)


@_traced("bitwise_nand", 2)
def bitwise_nand(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise NAND (NOT AND)."""
    return bitwise_not(bitwise_and(a, b, deps=deps), deps=deps)


@_traced("bitwise_nor", 2)
def bitwise_nor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise NOR (NOT OR)."""
    return bitwise_not(bitwise_or(a, b, deps=deps), deps=deps)


@_traced("bitwise_xnor", 2)
def bitwise_xnor(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Elementwise bitwise XNOR (NOT XOR)."""
    return bitwise_not(bitwise_xor(a, b, deps=deps), deps=deps)


@_traced("inv")
def inv(matrix: Any, *, deps: OpsDeps) -> Any:
    """The matrix inverse (NumPy's `linalg.inv`). The engine inverts every dtype, structure and
    view (LinearAlgebra.cpp `inverse`); a singular matrix raises LinAlgError."""
    return deps.native._inv(_engine_matrix(matrix, deps=deps))


@_traced("lstsq", 2)
def lstsq(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Least-squares solution x minimizing ||a @ x - b|| (minimum norm when rank-deficient).

    The engine runs NumPy's routine, LAPACK ?gelsd with NumPy's default cutoff, for every
    dtype, structure and view (it was pinv(a) @ b: the same answer, 1.5x slower; before that
    the normal equations, F-069).
    """
    native = (deps.native.MatrixBase, deps.native.VectorBase)
    a = a if isinstance(a, native) else _as_pycauset_array(deps.np_module.asarray(a), deps=deps)
    b = b if isinstance(b, native) else _as_pycauset_array(deps.np_module.asarray(b), deps=deps)
    result = deps.native.lstsq(a, b)
    return result


def slogdet(a: Any, *, deps: OpsDeps) -> tuple[Any, float]:
    """(sign, log|det|) from the engine's LU pivots, as NumPy: no overflow for large n, a
    complex unit sign for complex matrices, (0, -inf) when singular (F-070)."""
    return _engine_matrix(a, deps=deps)._slogdet()


@_traced("cond")
def cond(a: Any, *, deps: OpsDeps, p: Any = None) -> float:
    """Condition number, NumPy's definition (F-003): p None/2 is smax/smin, -2 is smin/smax
    (singular values from the engine's SVD); other p is ||A||_p * ||A^-1||_p. Singular: inf."""
    a = _engine_matrix(a, deps=deps)
    if p is None or p == 2 or p == -2:
        smax = deps.native._matrix_norm(a, "2")
        smin = deps.native._matrix_norm(a, "-2")
        if p == -2:
            return smin / smax if smax != 0 else float("inf")
        return smax / smin if smin != 0 else float("inf")
    try:
        inv_a = inv(a, deps=deps)
    except deps.np_module.linalg.LinAlgError:  # singular: NumPy returns inf here too
        return float("inf")
    return float(norm(a, p, deps=deps) * norm(inv_a, p, deps=deps))


def _engine_matrix(a: Any, *, deps: OpsDeps) -> Any:
    """A PyCauset matrix as is; anything else (NumPy array, nested lists) as a dense one.
    A BlockMatrix is not converted behind the caller's back (it may be far larger than RAM):
    the caller converts it explicitly."""
    if isinstance(a, deps.native.MatrixBase):
        return a
    from .blockmatrix import BlockMatrix

    if isinstance(a, BlockMatrix):
        raise TypeError(
            "this operation does not take a BlockMatrix; convert it explicitly with "
            "pc.matrix(np.asarray(block_matrix)) if it fits in memory"
        )
    return deps.Matrix(deps.np_module.asarray(a))


# Eigen decompositions, Cholesky, QR and pinv run in the engine (math/LinearAlgebra.cpp,
# compute/cpu/CpuLinalg.cpp) for every dtype, structure and view, in NumPy's LAPACK dtype;
# no NumPy fallback (F-090). eigh/eigvalsh/cholesky read the lower triangle, as NumPy does.

@_traced("eigh")
def eigh(a: Any, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Eigenvalues (real, ascending) and eigenvectors of a symmetric / Hermitian matrix."""
    return tuple(deps.native.eigh(_engine_matrix(a, deps=deps)))


@_traced("eigvalsh")
def eigvalsh(a: Any, *, deps: OpsDeps) -> Any:
    """Eigenvalues (real, ascending) of a symmetric / Hermitian matrix."""
    return deps.native.eigvalsh(_engine_matrix(a, deps=deps))


@_traced("eig")
def eig(a: Any, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Complex eigenvalues and right eigenvectors of a general square matrix."""
    return tuple(deps.native.eig(_engine_matrix(a, deps=deps)))


@_traced("eigvals")
def eigvals(a: Any, *, deps: OpsDeps) -> Any:
    """Complex eigenvalues of a general square matrix."""
    return deps.native.eigvals(_engine_matrix(a, deps=deps))


@_traced("eigvals_arnoldi")
def eigvals_arnoldi(a: Any, k: int, m: int, tol: float, *, deps: OpsDeps) -> Any:
    """Top-k eigenvalues by magnitude via restarted Lanczos/Arnoldi (native, Spectra).

    Symmetric input returns real (float64) eigenvalues, otherwise complex128. The
    matrix is only touched through matrix-vector products, so disk-backed matrices
    work without materializing. Raises RuntimeError if the iteration does not converge
    to `tol` (increase `m`). `m` is the Krylov subspace size (raised to at least 2k+1).
    """
    return deps.native.eigvals_arnoldi(a, int(k), int(m), float(tol))


def eigvals_skew(a: Any, k: int, *, deps: OpsDeps) -> Any:
    """Top-k (by magnitude) eigenvalues of a real skew-symmetric matrix.

    A real skew-symmetric matrix (A == -A.T) has purely imaginary eigenvalues
    that come in +/-i*lambda pairs (plus a zero eigenvalue for odd dimension).
    Engine only (LAPACK on the skew-symmetric tridiagonal form); no NumPy fallback.
    """

    return deps.native.eigvals_skew(_engine_matrix(a, deps=deps), k)


def eig_skew(a: Any, k: int, *, deps: OpsDeps) -> tuple[Any, Any]:
    """Top-k eigenvalues + eigenvectors of a real skew-symmetric matrix.

    A real skew-symmetric matrix (A == -A.T) has purely imaginary eigenvalues in
    +/-i*lambda pairs. Returns ``(w, v)`` where ``w`` holds the top-|k| eigenvalues
    by magnitude and ``v`` is the n x min(k, n) matrix of the matching (complex)
    eigenvectors, sorted the same way. This is the diagonalization the Sorkin-Johnston
    W = positive part of i*Delta needs.
    """

    return tuple(deps.native.eig_skew(_engine_matrix(a, deps=deps), k))


def solve_triangular(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Solve A X = B for a triangular, diagonal or identity A (by class: TriangularMatrix and
    its transpose, DiagonalMatrix, IdentityMatrix); anything else raises ValueError.

    The engine's solve handles every structure, view and dtype (F-068). It used to accept
    any matrix whose `is_*_triangular` property was set, and not a TriangularMatrix (F-092).
    """
    if not isinstance(a, deps.native.MatrixBase) or deps.native._structure_of(a) not in (
        "triangular", "diagonal", "identity"
    ):
        raise ValueError("solve_triangular: a must be a TriangularMatrix, DiagonalMatrix or IdentityMatrix")
    return solve(a, b, deps=deps)


@_traced("pinv")
def pinv(a: Any, *, deps: OpsDeps) -> Any:
    """Moore-Penrose pseudoinverse, as NumPy's pinv: A = U S Vᴴ → A⁺ = V S⁺ Uᴴ, singular values
    at or below max(m, n) · eps · max(s) treated as zero (F-079, F-090)."""
    return deps.native.pinv(_engine_matrix(a, deps=deps))


@_traced("trace")
def trace(a: Any, *, deps: OpsDeps) -> Any:
    """Sum of the diagonal: exact int for integer matrices, float, or complex (F-059)."""
    return _engine_matrix(a, deps=deps).trace()


@_traced("determinant")
def determinant(a: Any, *, deps: OpsDeps) -> Any:
    """Determinant of a square matrix: complex for complex matrices, else float (F-059)."""
    return _engine_matrix(a, deps=deps)._determinant()


def _ord_name(ord: Any) -> str:
    if isinstance(ord, str):
        return ord
    if ord in (float("inf"), float("-inf")):
        return "inf" if ord > 0 else "-inf"
    if ord in (1, -1, 2, -2):
        return str(int(ord))
    raise ValueError(f"norm: invalid norm order {ord!r} for matrices")


@_traced("norm")
def norm(x: Any, ord: Any = None, *, deps: OpsDeps) -> float:
    """Matrix or vector norm with NumPy's orders, in the engine.

    Matrices: None/'fro', 'nuc', 1, -1, 2, -2, inf, -inf. Vectors: None (2-norm), any real p,
    inf, -inf, 0 (count of non-zeros).
    """
    if isinstance(x, deps.native.VectorBase):
        return float(deps.native.norm(x) if ord is None else deps.native._vector_norm(x, float(ord)))
    x = _engine_matrix(x, deps=deps)
    if ord is None or ord == "fro":
        return float(deps.native.norm(x))
    return float(deps.native._matrix_norm(x, _ord_name(ord)))


@_traced("svdvals")
def svdvals(a: Any, *, deps: OpsDeps) -> Any:
    """Singular values of a matrix (the S vector of an SVD), descending; engine SVD (F-079)."""
    return svd(a, compute_uv=False, deps=deps)


@_traced("matrix_rank")
def matrix_rank(a: Any, tol: Any = None, *, deps: OpsDeps) -> int:
    """Numerical rank: singular values above tol (NumPy's default tol when None), from the
    engine's SVD. The structural shortcuts that were here counted the non-zero diagonal of a
    triangular matrix (wrong: a strictly triangular matrix of ones has rank n - 1) and
    trusted user flags (F-090)."""
    return int(deps.native.matrix_rank(_engine_matrix(a, deps=deps), -1.0 if tol is None else float(tol)))


@_traced("matrix_power")
def matrix_power(a: Any, n: int, *, deps: OpsDeps) -> Any:
    """Integer power of a square matrix (A^n) via binary exponentiation; n < 0 inverts first."""
    shape = _safe_rows_cols(a)
    if shape is not None and shape[0] != shape[1]:
        raise ValueError("matrix_power requires a square matrix")

    if n == 0:
        return _identity_like(a, shape, deps)
    if n == 1:
        return a

    # No identity shortcut: a scaled identity (2*I)^n is 2^n*I, and identity x identity is
    # already O(1) in the engine (F-067).
    # No is_zero shortcut: results come from the values (F-090).
    if n < 0:
        return matrix_power(inv(a, deps=deps), -n, deps=deps)

    # Binary exponentiation via matmul. No identity seed: multiplying by a float64
    # identity would promote integer/causal powers to float (and costs one extra matmul).
    result = None
    base = a
    while n > 0:
        if n & 1:
            result = base if result is None else matmul(result, base, deps=deps)
        n >>= 1
        if n:
            base = matmul(base, base, deps=deps)
    return result


def _identity_like(a: Any, shape: Any, deps: OpsDeps) -> Any:
    """a^0: the identity in a's dtype, as NumPy's matrix_power (an int16 matrix gives an int16
    identity; it gave float64 before)."""
    return deps.native.IdentityMatrix(shape[0], dtype=str(a.dtype))


def _engine_vector(x: Any, *, deps: OpsDeps) -> Any:
    """A PyCauset vector as is; an array-like flattened into one (NumPy's ravel). A PyCauset
    matrix raises: outer/cross/vecdot take vectors."""
    if isinstance(x, deps.native.VectorBase):
        return x
    if isinstance(x, deps.native.MatrixBase):
        raise TypeError("expected a vector, got a matrix")
    import pycauset  # the public constructor (every dtype); the package is loaded by now

    return pycauset.vector(deps.np_module.ravel(deps.np_module.asarray(x)))


@_traced("outer", 2)
def outer(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Outer product out[i, j] = a[i] * b[j], as NumPy's outer (any lengths, multiply dtype)."""
    return deps.native.outer(_engine_vector(a, deps=deps), _engine_vector(b, deps=deps))


@_traced("cross", 2)
def cross(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Cross product of two 3-element vectors, as NumPy's cross (multiply dtype; bool raises)."""
    return deps.native.cross(_engine_vector(a, deps=deps), _engine_vector(b, deps=deps))


@_traced("vecdot", 2)
def vecdot(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Conjugate dot product sum(conj(a) * b), as NumPy's vdot; for real input this is dot."""
    return _engine_vector(a, deps=deps).conj().dot(_engine_vector(b, deps=deps))


@_traced("cholesky")
def cholesky(a: Any, *, deps: OpsDeps) -> Any:
    """Lower Cholesky factor L, A = L Lᴴ (reads the lower triangle, as NumPy). Not positive
    definite: ValueError (NumPy's LinAlgError is a ValueError)."""
    return deps.native.cholesky(_engine_matrix(a, deps=deps))


@_traced("qr")
def qr(a: Any, mode: str = 'reduced', *, deps: OpsDeps) -> Any:
    """QR decomposition with NumPy's modes 'reduced' (Q m×k, R k×n), 'complete' (Q m×m, R m×n)
    and 'r' (R only). NumPy's 'raw' (LAPACK's Householder form) is not offered."""
    if mode not in ("reduced", "complete", "r"):
        raise ValueError(f"qr: mode must be 'reduced', 'complete' or 'r' (got {mode!r})")
    q, r = deps.native.qr(_engine_matrix(a, deps=deps), mode == "complete")
    if mode == "r":
        return r
    return (q, r)


@_traced("svd")
def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True, *, deps: OpsDeps) -> Any:
    """SVD with NumPy's signature and shapes, computed by the engine (?gesdd) for every dtype,
    structure and view; no NumPy fallback (F-079)."""
    if not isinstance(a, deps.native.MatrixBase):
        a = deps.Matrix(deps.np_module.asarray(a))
    u, s, vt = deps.native.svd(a, bool(full_matrices), bool(compute_uv))
    if not compute_uv:  # singular values only (?gesdd job 'N')
        return s
    return u, s, vt


@_traced("lu")
def lu(a: Any, *, deps: OpsDeps) -> Any:
    """(P, L, U) with A = P L U, as SciPy's lu: P real (m×m), L unit lower (m×k), U (k×n)."""
    return tuple(deps.native.lu(_engine_matrix(a, deps=deps)))


@_traced("solve", 2)
def solve(a: Any, b: Any, *, deps: OpsDeps) -> Any:
    """Solve AX = B (B a matrix or a vector) in the engine: any structure, view and dtype.

    The engine writes structured/viewed operands out and runs LAPACK ?gesv (F-068). There is
    no NumPy fallback: an engine error is the answer.
    """
    native = (deps.native.MatrixBase, deps.native.VectorBase)
    a = a if isinstance(a, native) else _as_pycauset_array(deps.np_module.asarray(a), deps=deps)
    b = b if isinstance(b, native) else _as_pycauset_array(deps.np_module.asarray(b), deps=deps)
    x = deps.native.solve(a, b)
    return x
