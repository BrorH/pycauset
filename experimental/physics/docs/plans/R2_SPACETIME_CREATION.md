# R2 Spacetime Creation, Recipe, Protocol & Tooling (Spec)

**Status**: Planning. Defines the full "easy ladder" for defining custom spacetimes.
**Companion**: `R2_PLAN_MAP.md` (§5), `R2_API_DESIGN.md` (§2), `R2_SPACETIME_LIBRARY.md`.

---

## 1. The minimal `Spacetime` contract

Every spacetime, library or custom, implements this. It is the *only* thing the sprinkler and
the field engine are allowed to depend on.

```python
class Spacetime(ABC):
    dimension: int                # total d
    signature: tuple[int, int]    # (t, s) = (timelike, spacelike); Lorentzian = (1, d-1)

    def sample(self, rng, n) -> "ndarray (n, d)":
        """Draw n points, uniform w.r.t. the measure whose mass is volume()."""

    def is_causal(self, u, v) -> bool:
        """Strict partial order, TRANSITIVE (the closure, not links)."""

    def volume(self) -> float:
        """Total mass of the sampling measure; 0 < volume() < inf."""

    # --- optional ---
    def is_causal_batch(self, coords) -> "ndarray (n, n) bool": ...   # Rung 2 (fast)
    def scalar_coeffs(self, mass, density) -> "(a, b)": ...           # else NotImplementedError
    def to_embedding(self, coords) -> "ndarray": ...                 # presentation (identity default)
    def boundary(self) -> "list[ndarray]": ...                       # presentation (empty default)
```

**RNG contract (hard requirement):** `sample` receives a seeded RNG object and must be a *pure
function* of it, no global random state. This is what makes
`same (spacetime, seed, n) ⇒ bit-identical points and order` hold, which the hybrid coordinate
model depends on.

**Signature contract:** `is_causal` is only meaningful when `signature == (1, d-1)`. The base class
provides a default that raises `NotImplementedError` for `t != 1` (Euclidean ⇒ point process, not a
causet). Multi-time (`t > 1`) spacetimes may override it if they define a "future" convention.

---

## 2. The easy ladder

### Rung 0, `spacetime.create(recipe)` (declarative, no class)

```python
st = spacetime.create(
    dimension=3,
    signature=(1, 2),       # optional; default (1, d-1) Lorentzian, a documented default, not inference
    domain="box",           # box | diamond | cylinder | ball | slab | none
    metric="flat",          # flat | de_sitter | anti_de_sitter | flrw
    time_extent=4.0,
    space_extent=(2.0, 2.0),
    # metric/domain-specific params only where the chosen template declares them
)
```

**Recipe schema (spec):**

| Field | Type | Default | Rules |
| :-- | :-- | :-- | :-- |
| `name` | `str` | required for persistence | registry key; unique |
| `dimension` | `int ≥ 2` | required | total d |
| `signature` | `(t, s)` | `(1, d-1)` | must satisfy `t + s == dimension` |
| `domain` | enum | required | `box`/`diamond`/`cylinder`/`ball`/`slab`/`none` |
| `metric` | enum | required | `flat`/`de_sitter`/`anti_de_sitter`/`flrw` |
| domain params |, | per-domain | e.g. `time_extent`, `space_extent`, `height`, `circumference`, `radius` |
| metric params |, | per-metric | e.g. `curvature_radius`, `scale_factor`, `k` |
| `periodic` | `bool \| list[int]` | `False` | which spatial dims wrap |

**Mapping rule:** `(domain, metric)` selects a registered **template**. Each template declares
exactly which parameters it *requires* and which it *forbids*. Missing-required or
not-supported-combination ⇒ an immediate error listing the valid combinations. **No silent
defaults for physics parameters**, the only defaults are `signature` (documented) and `periodic`
(documented), neither of which is inference.

**Result:** a configured `Spacetime` instance (implementation detail: a built-in class, a composed
decorator stack, or a generated subclass, but the user only ever sees a `Spacetime`).

### Rung 1, subclass (three methods)

The §1 contract written by hand. This is the escape hatch and the pedagogical path; `create` is
implemented *on top of* it, so nothing is magic.

```python
@spacetime.register("my_diamond_4d")
class MyDiamond4D(spacetime.Spacetime):
    dimension = 4
    signature = (1, 3)

    def sample(self, rng, n):
        return rng.uniform(0, 1, size=(n, 4))

    def is_causal(self, u, v):
        return all(u[i] < v[i] for i in range(4))

    def volume(self):
        return 1.0
```

### Rung 2, batch hook (optional speed)

Add `is_causal_batch(coords) -> (n, n) bool` (upper-triangular) and the sprinkler runs the O(n²)
pairwise step in NumPy/C instead of a Python loop. If absent, the sprinkler falls back to calling
`is_causal(u, v)` per pair. Sampling is already vectorized (`sample` returns `(n, d)`), so the
"tier" distinction is about *causality* only.

---

## 3. Modifying a library spacetime (composition + subclass)

Two sanctioned ways (director decision #6):

1. **Subclass** a library spacetime and override one method.
2. **Compose** with thin decorators:

| Decorator | Effect | Recomputes |
| :-- | :-- | :-- |
| `RestrictedSpacetime(base, region)` | keep a subregion | `volume`, `sample` (rejection), `is_causal` (inherits) |
| `ConformalSpacetime(base, Ω)` | multiply metric by conformal factor | `sample` (importance/rejection); lightcone unchanged |
| `TransformedSpacetime(base, f)` | apply coordinate map | `sample`, `is_causal`, `volume` |
| `PeriodicSpacetime(base, dims)` | wrap spatial dims | `is_causal` (wrap distance) |

Decorators compose (`Restricted(Conformal(DeSitter(d), Ω), region)`) and **must keep `volume()`
consistent with the wrapped `sample()`**, that consistency is an invariant the runtime checks on
construction where feasible.

---

## 4. Code generation + the online tool

`spacetime.export_python(recipe_or_st) -> str` emits a **paste-ready Rung-1 subclass** generated
from the *same template* `create()` uses. The online generator is a thin front-end over this
template, so it can never drift:

```
[web form] ──(1:1 fields)──> recipe ──> export_python template ──> code string
```

- **No guessing in the tool:** every box maps to an explicit parameter; fields we can't safely
  default are left blank and *required*, with an inline note explaining why.
- **Edit existing spacetimes:** the studio opens with a *library* picker (Diamond, Cylinder, Box,
  dS, AdS, FLRW, black holes…) that pre-fills the form; the user edits from there and the recipe +
  code update live. "Blank / custom" is just the empty preset.

### 4.1 Hosting & launching (the "fun" delivery)

Two launch modes, both pure client-side (no backend needed for R2.0):

1. **Hosted (GitHub Pages):** publish the studio as a static page under the docs site (the repo
   already deploys MkDocs to GitHub Pages). Best for discoverability and sharing.
2. **Open-in-browser, Plotly-style:** `pycauset.studio()` opens a self-contained HTML file in the
   default browser, exactly how Plotly's `fig.show()` opens a local page. Since the studio is
   pure JS, no server is required. `pycauset.studio(hosted=True)` opens the GitHub Pages URL
   instead.

Creative note: for R2.0 the studio is a *generator* (form → recipe → code) and can be fully
static. A later "live preview" (actually sprinkle and render the causet in the browser) needs the
pycauset engine behind a tiny local server, the Plotly model again, and is an R2.x stretch, not
R2.0.

---

## 5. Persistence (save/load spacetimes)

- **Registry:** `@spacetime.register("name")` + built-in names. `name` is the persistence key.
- **Save** serializes the **recipe**, not code: `{kind, name, params, transforms}`. A modified
  `DeSitter` saves as `{kind: "de_sitter", params: {...}, transforms: [{op: "conformal", Ω: ...}]}`.
- **Load** looks up `kind`/`name` in the registry; if absent, raise
  `UnknownSpacetime("... is not registered; import the module that defines it")`.
- **Collision policy (agreed):** explicit error on duplicate name, with an `overwrite=True`
  override, no silent last-wins.

---

## 6. Acceptance criteria

- **Rung 0:** `create` covers every library `(domain, metric)` pair it advertises; unsupported
  combos raise with a "valid options" message; `export_python(recipe)` round-trips to an equivalent
  `Spacetime`.
- **Rung 1:** a 3-method subclass sprinkles, validates, and visualizes with no other boilerplate.
- **Rung 2:** `is_causal_batch` present ⇒ sprinkler uses it; absent ⇒ correct fallback; both paths
  give identical orders for the same seed.
- **Composition:** decorators preserve `volume ↔ sample` consistency; `Restricted`/`Periodic`
  produce orders identical to an equivalent hand-written spacetime.
- **Persistence:** library + custom + modified spacetimes all round-trip save/load; unknown names
  raise a clear error.
