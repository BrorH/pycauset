"""Synthetic poset generators (R2_SYNTH) — "a causet is just a poset".

These build valid causal orders directly (no continuum geometry), for testing,
pedagogy, and null models. Every generator returns a `CausalSet` whose order passes
`validate()` (reflexive-free, antisymmetric, transitive).
"""

from __future__ import annotations

import numpy as np

from .causet import CausalSet


def _transitive_closure(C: np.ndarray) -> np.ndarray:
    """Reflexive-free transitive closure of an upper-triangular boolean matrix."""
    out = np.asarray(C, dtype=bool).copy()
    n = out.shape[0]
    for k in range(n):
        out |= out[:, k, None] & out[None, k, :]
    return np.triu(out, 1)


def _causet(C: np.ndarray) -> CausalSet:
    """Wrap a dense (n, n) upper-triangular boolean order in a validated CausalSet."""
    import pycauset as pc

    C = np.asarray(C, dtype=bool)
    n = C.shape[0]
    return CausalSet(n=n, matrix=pc.causal_matrix(C), validate=True)


def chain(n: int) -> CausalSet:
    """The total order ``0 < 1 < ... < n-1``."""
    return _causet(np.triu(np.ones((n, n), dtype=bool), 1))


def antichain(n: int) -> CausalSet:
    """The empty order (no relations)."""
    return _causet(np.zeros((n, n), dtype=bool))


def _random_dag(p: float, n: int, seed) -> CausalSet:
    rng = np.random.default_rng(seed)
    C = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                C[i, j] = True
    return _causet(_transitive_closure(C))


def transitive_percolation(p: float, n: int, seed=None) -> CausalSet:
    """Random causet from bond percolation on a total order (the classical model)."""
    return _random_dag(p, n, seed)


def random_dag_order(p: float, n: int, seed=None) -> CausalSet:
    """Random acyclic upper-triangular edges + transitive closure (the raw random causet)."""
    return _random_dag(p, n, seed)


def product_order(dims) -> CausalSet:
    """The grid poset (product of chains)."""
    dims = tuple(int(d) for d in dims)
    n = int(np.prod(dims))
    coords = np.array(np.unravel_index(np.arange(n), dims)).T
    C = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if np.all(coords[i] <= coords[j]) and np.any(coords[i] < coords[j]):
                C[i, j] = True
    return _causet(C)


def poset(relations, n=None) -> CausalSet:
    """An explicit user order from a list of ``(i, j)`` pairs (transitively closed)."""
    relations = list(relations)
    if n is None:
        n = max((max(pair) for pair in relations), default=-1) + 1
    C = np.zeros((n, n), dtype=bool)
    for i, j in relations:
        if 0 <= i < n and 0 <= j < n and i < j:
            C[i, j] = True
    return _causet(_transitive_closure(C))


__all__ = [
    "chain",
    "antichain",
    "transitive_percolation",
    "random_dag_order",
    "product_order",
    "poset",
]
