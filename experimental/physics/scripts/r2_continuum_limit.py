"""R2_QA / R2_CMVP, continuum-limit benchmark (massless 1+1, Minkowski diamond).

Reproduces with::

    python benchmarks/r2_continuum_limit.py

For the massless 1+1 field the discrete Pauli–Jordan ``iΔ = (i/2)(C − Cᵀ)`` is
**exact** against the continuum ``iΔ = (i/2) sgn(Δt) θ(σ)`` sampled at the causet's
points, there is no discretization error, so this also pins the sign/scale
convention (R2_CONV). What *does* converge is the Sorkin–Johnston Wightman
``W = positive part of iΔ``: as the number of sprinkled points grows, its
finite-dimensional positive spectrum approaches the continuum positive-frequency
spectrum. This benchmark reports both:

* ``max |iΔ_discrete − iΔ_continuum|``, the convention pin (should be ~0 at every n).
* the SJ Wightman positive spectrum (count, extremal eigenvalues, trace) vs ``n``.

The analytic continuum **Wightman** (massive/Bessel + the massless log) is a
deferred R2_CMVP item, see `documentation/project/plans/R2_ROADMAP.md`.
"""

from __future__ import annotations

import numpy as np

import pycauset as pc


def run(n: int, seed: int = 11):
    st = pc.spacetime.MinkowskiDiamond(2)
    c = pc.CausalSet(n=n, spacetime=st, seed=seed)

    phi = pc.field("scalar", mass=0.0)
    q_c = phi.on(c)
    q_ct = phi.on(st)

    coords = st.to_embedding(c.embedding)  # lightcone (u,v) -> physical (t,x)

    iD_disc = np.asarray(q_c.pauli_jordan())
    iD_cont = np.asarray(q_ct.at(coords, which="pauli_jordan"))
    pin_err = float(np.max(np.abs(iD_disc - iD_cont)))

    # SJ Wightman = positive-eigenvalue part of iΔ (purely imaginary spectrum).
    w = np.asarray(q_c.wightman())
    w_eig = np.linalg.eigvalsh(w)
    pos = w_eig[w_eig > 1e-12]

    return {
        "n": n,
        "pin_err": pin_err,
        "n_pos_modes": int(pos.size),
        "w_min_pos": float(pos.min()) if pos.size else float("nan"),
        "w_max_pos": float(pos.max()) if pos.size else float("nan"),
        "w_trace": float(np.real(np.trace(w))),
    }


def main() -> None:
    ns = [32, 64, 128, 256, 512, 1024]
    print(f"{'n':>6} {'|iD err|':>12} {'#W+modes':>10} {'W min+':>14} {'W max+':>14} {'tr(W)':>14}")
    print("-" * 74)
    for n in ns:
        r = run(n)
        print(
            f"{r['n']:>6} {r['pin_err']:>12.2e} {r['n_pos_modes']:>10} "
            f"{r['w_min_pos']:>14.6f} {r['w_max_pos']:>14.6f} {r['w_trace']:>14.6f}"
        )
    print("-" * 74)
    print("|iD err| stays ~machine-epsilon at every n (the massless 1+1 convention is pinned).")
    print("The SJ Wightman positive spectrum grows with n (continuum limit of W).")


if __name__ == "__main__":
    main()
