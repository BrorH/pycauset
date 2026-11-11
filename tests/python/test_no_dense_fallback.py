"""Physics operations must not fall back to dense NumPy (BUG_LOG 2026-11-11, Critical).

PyCauset exists so that causal sets bigger than NumPy can handle stay in the
bit-packed, disk-spillable engine. Every operation below currently converts the
whole n x n causal matrix (or propagator) to a dense NumPy array, which makes it
O(n^2) memory and, at 20,000 elements, ~100 s per call.

The detector: with NumPy export disabled (`set_export_max_bytes(1)`), each
operation must still work. Genuine exports, `pycauset.to_numpy(...)` and
`CausalSet.coordinates()` (an n x d array), are out of scope.

Every case is a strict xfail today. When an operation is moved into the engine,
its case starts passing, strict mode turns that into a failure, and the marker
must be removed: that is the point, it keeps the scoreboard honest.
"""

import warnings

import pycauset as pc
import pytest
from pycauset import synthetic

DENSE_FALLBACK = pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="dense NumPy fallback, see tests/BUG_LOG.md (2026-11-11)",
)

N = 60


@pytest.fixture
def causet():
    return pc.CausalSet(n=N, seed=7)


@pytest.fixture
def no_numpy_export():
    """Block every NumPy materialization of an engine object for the test body."""
    previous = pc._internal.export_guard.get_max_bytes()
    pc.set_export_max_bytes(1)
    try:
        yield
    finally:
        pc.set_export_max_bytes(previous)


def test_detector_blocks_dense_export(causet, no_numpy_export):
    """Sanity check: the detector really blocks a dense conversion."""
    import numpy as np

    with pytest.raises(RuntimeError):
        np.asarray(causet.C, dtype=bool)


def test_genuine_exports_are_exempt(causet, no_numpy_export):
    """coordinates() is an n x d result, not a dense n x n fallback."""
    assert causet.coordinates().shape == (N, 2)


CAUSET_OPS = {
    "relation_fraction": lambda c: c.relation_fraction(),
    "myrheim_meyer_dimension": lambda c: c.myrheim_meyer_dimension(),
    "past": lambda c: c.past(N // 2),
    "future": lambda c: c.future(N // 2),
    "interval": lambda c: c.interval(0, N - 1),
    "is_chain": lambda c: c.is_chain([0, 1, 2]),
    "is_antichain": lambda c: c.is_antichain([0, 1, 2]),
    "longest_chain": lambda c: c.longest_chain(),
    "layers": lambda c: c.layers(),
    "links": lambda c: c.links(),
    "validate": lambda c: c.validate(),
    # Plots draw a small subset but currently densify the whole matrix first.
    "plot_hasse_subset": lambda c: c.plot_hasse(max_points=10),
    "plot_causal_matrix_subset": lambda c: c.plot_causal_matrix(max_points=10),
    "plot_embedding_relations": lambda c: c.plot_embedding(show_relations=True),
}


@DENSE_FALLBACK
@pytest.mark.parametrize("op", sorted(CAUSET_OPS))
def test_causet_op_stays_native(op, causet, no_numpy_export):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CAUSET_OPS[op](causet)


FIELD_OPS = {
    "retarded": lambda f: f.retarded(),
    "pauli_jordan": lambda f: f.pauli_jordan(),
    "wightman": lambda f: f.wightman(),
    "entanglement_entropy": lambda f: f.entanglement_entropy(range(N // 2)),
}


@DENSE_FALLBACK
@pytest.mark.parametrize("op", sorted(FIELD_OPS))
def test_field_op_stays_native(op, causet, no_numpy_export):
    field = pc.field("scalar", mass=1.0).on(causet)
    FIELD_OPS[op](field)


SYNTHETIC_OPS = {
    "chain": lambda: synthetic.chain(N),
    "antichain": lambda: synthetic.antichain(N),
    "transitive_percolation": lambda: synthetic.transitive_percolation(0.1, N, seed=1),
    "random_dag_order": lambda: synthetic.random_dag_order(0.1, N, seed=1),
    "product_order": lambda: synthetic.product_order((6, 10)),
    "poset": lambda: synthetic.poset([(0, 1), (1, 2)], n=N),
}


@DENSE_FALLBACK
@pytest.mark.parametrize("op", sorted(SYNTHETIC_OPS))
def test_synthetic_generator_stays_native(op, no_numpy_export):
    SYNTHETIC_OPS[op]()
