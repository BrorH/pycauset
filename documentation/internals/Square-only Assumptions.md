# Square-only assumptions (NxN)

Dense matrices of every dtype are rectangular (rows x cols), bit-packed `bool` included.
What is still square is square by definition.

## Square by structure

Triangular (causal matrices included), symmetric, antisymmetric, Hermitian,
skew-Hermitian and diagonal matrices are square: `ObjectFactory` rejects `rows != cols`
for them, and so do their Python constructors. Identity matrices are shape-flexible:
`pc.identity([rows, cols])` has ones on the diagonal up to `min(rows, cols)`.

## Square by mathematics

`determinant`, `slogdet`, `inv`, `solve` (its `A`), `cholesky`, `eig`, `eigh` and
the other eigen problems, `matrix_power`.
They raise `ValueError` for a rectangular operand. `svd`, `qr`, `lu`, `pinv`, `lstsq`,
`matrix_rank`, `norm` and `trace` take rectangular matrices, as in NumPy.

## Where rectangular shapes are carried

`MatrixBase` / `PersistentObject` store `rows` and `cols`; `ObjectFactory::create_matrix`
takes both; the container stores both; `shape` in Python is `(rows, cols)`. See
[[guides/NxM Support|NxM Support]].
