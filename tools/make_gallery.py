"""Regenerate the plot images the guides show (documentation/docs/assets/gallery/).

    python tools/make_gallery.py        # needs plotly and kaleido (kaleido needs a Chrome)

Public API only: when this script stops running, the images in the guides are out of date.
Analyzing a Causal Set.md and Visualization.md use the four images written here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pycauset as pc

OUT = Path(__file__).resolve().parents[1] / "documentation" / "docs" / "assets" / "gallery"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diamond2 = pc.spacetime.MinkowskiDiamond(2)

    # Visualization.md: a 2D diamond, 3000 points.
    c = pc.causet(n=3000, spacetime=diamond2, seed=42)
    pc.plot_embedding(c, title="Minkowski diamond, 3000 points").write_image(OUT / "diamond_embedding.png", scale=2)

    # Visualization.md: a 4D diamond shows its first three axes (and warns that it does).
    c4 = pc.causet(n=2000, spacetime=pc.spacetime.MinkowskiDiamond(4), seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pc.PyCausetWarning)  # "rendering the first 3 axes"
        fig = pc.plot_embedding(c4, title="4D diamond, 2000 points (first three axes)")
    fig.write_image(OUT / "diamond_4d_embedding.png", scale=2)

    # Visualization.md: the Hasse diagram (links) of a small diamond.
    c_small = pc.causet(n=80, spacetime=diamond2, seed=7)
    pc.plot_hasse(c_small, title="Hasse diagram of a small diamond, 80 points").write_image(
        OUT / "diamond_hasse.png", scale=2)

    # Analyzing a Causal Set.md, Visualization.md: the causal matrix as a heatmap.
    c_heat = pc.causet(n=120, spacetime=diamond2, seed=7)
    fig = pc.plot_causal_matrix(c_heat, title="Causal matrix of a 120-point diamond")
    fig.update_layout(width=720, height=720)
    fig.write_image(OUT / "causal_matrix.png", scale=2)

    print("wrote:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
