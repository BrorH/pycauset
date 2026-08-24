"""Render PyCauset vs NumPy benchmark graphs from benchmarks/results.json.

Run `python benchmarks/bench.py` first, then `python benchmarks/plot.py`.
Output PNGs are written to documentation/docs/assets/benchmarks/.
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = Path("documentation/docs/assets/benchmarks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"numpy": "#1f77b4", "pycauset": "#ff7f0e"}


def _load() -> dict:
    with open("benchmarks/results.json") as f:
        return json.load(f)


def _time_vs_n(results: dict, ops: list[str], title: str, filename: str) -> None:
    fig = make_subplots(rows=1, cols=len(ops), shared_yaxes=True,
                        subplot_titles=ops, horizontal_spacing=0.06)
    for i, op in enumerate(ops):
        r = results[op]
        sizes = r["sizes"]
        fig.add_trace(go.Scatter(x=sizes, y=r["numpy_ms"], name="NumPy",
                                 mode="lines+markers", line=dict(color=COLORS["numpy"]),
                                 legendgroup="numpy", showlegend=(i == 0)),
                      row=1, col=i + 1)
        fig.add_trace(go.Scatter(x=sizes, y=r["pycauset_ms"], name="PyCauset",
                                 mode="lines+markers", line=dict(color=COLORS["pycauset"]),
                                 legendgroup="pycauset", showlegend=(i == 0)),
                      row=1, col=i + 1)
    fig.update_layout(title=title, height=420, width=360 * len(ops),
                      yaxis_type="log", xaxis_type="log",
                      yaxis_title="time (ms)", xaxis_title="n",
                      template="plotly_white")
    fig.write_image(str(OUT_DIR / filename), scale=2)


def _speedup_bar(results: dict, filename: str) -> None:
    ops = ["matmul", "inverse", "solve", "cholesky", "svd", "eigh", "eigvalsh", "add", "dot"]
    labels = []
    speedups = []
    for op in ops:
        r = results[op]
        # speedup at the largest measured size (numpy / pycauset; >1 means faster)
        labels.append(op)
        speedups.append(r["numpy_ms"][-1] / r["pycauset_ms"][-1])
    fig = go.Figure(go.Bar(x=labels, y=speedups, text=[f"{s:.2f}x" for s in speedups],
                           textposition="outside", marker_color=COLORS["pycauset"]))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="NumPy parity (1.0x)")
    fig.update_layout(title="PyCauset speedup vs NumPy (largest size, >1 means faster)",
                      yaxis_title="speedup (numpy / pycauset)", template="plotly_white",
                      height=460, width=820)
    fig.write_image(str(OUT_DIR / filename), scale=2)


def main() -> None:
    results = _load()
    _time_vs_n(results, ["matmul", "inverse", "solve"], "Time vs n (log-log)", "time_matmul_fact.png")
    _time_vs_n(results, ["cholesky", "eigh", "svd"], "Time vs n (log-log)", "time_eigen_svd.png")
    _time_vs_n(results, ["add", "dot"], "Time vs n (log-log)", "time_elem_dot.png")
    _speedup_bar(results, "speedup_by_op.png")
    print("Wrote PNGs to", OUT_DIR)


if __name__ == "__main__":
    main()
