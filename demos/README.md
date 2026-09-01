# Demos

Runnable scripts that show what PyCauset does, one idea at a time. Each one is
self-contained: run it from the repo root with `python demos/<name>.py`.

Plot images land in `demos/output/` (requires `kaleido` for static PNGs; without it
the script prints the results and skips the image).

| Script | What it shows |
| :--- | :--- |
| `01_hello_causet.py` | Sprinkle a causal set, inspect it, plot it, save/load it. |
| `02_dimension_from_order.py` | Recover the spacetime dimension from the order alone (Myrheim-Meyer). |
| `03_scalar_field.py` | A scalar field: propagators, the Sorkin-Johnston vacuum, entanglement. |
| `04_spacetimes.py` | The built-in spacetime library, plus a custom one. |
| `05_matrix_engine.py` | The matrix/vector engine on its own, and synthetic orders. |
| `06_large_n.py` | 150,000 points whose causal matrix spills to disk (~1-2 min). |

Run them all in order:

```bash
pip install pycauset kaleido
python demos/01_hello_causet.py
python demos/02_dimension_from_order.py
python demos/03_scalar_field.py
python demos/04_spacetimes.py
python demos/05_matrix_engine.py
python demos/06_large_n.py
```

`06_large_n.py` takes a minute or two, so start it early in its own terminal and
come back to it — it prints progress as it sprinkles, then saves a plot and announces
itself with a `DONE` banner when it finishes.
