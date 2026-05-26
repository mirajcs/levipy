# LeviPy

A pure-Python, [SymPy](https://www.sympy.org/)-backed library for **symbolic
differential geometry** and **general relativity** computations — the
Levi-Civita connection, curvature tensors, geodesics, and more, all computed
exactly as closed-form symbolic expressions.

What sets LeviPy apart is its **Lean 4 bridge**: it doesn't just *compute*
geometric quantities, it can export them as **machine-checkable proof
obligations** in [Lean 4](https://leanprover.github.io/) against
[Mathlib](https://leanprover-community.github.io/mathlib4_docs/) — turning a
symbolic curvature calculation into a formal theorem a proof assistant can
verify. See [Lean 4 formal verification](#lean-4-formal-verification-bridge).

## Features

- **Metric tensor** `g_{ij}` with automatic inverse `g^{ij}`
- **Christoffel symbols** `Γ^k_{ij}` of the Levi-Civita connection
- **Riemann curvature** tensor `R^l_{ijk}`
- **Ricci tensor** `R_{ij}` and **Ricci scalar** `R`
- **Einstein tensor** `G_{ij}`
- **Geodesic equations** for an affinely parametrised curve
- **Parallel transport** of a vector field along a curve
- **Covariant derivatives** of vectors, covectors, and (1,1)-tensors
- **Lie bracket** `[X, Y]` of two vector fields
- **Lean 4 bridge** — translate SymPy expressions to Lean 4 terms, emit
  theorem stubs, and link to relevant [Mathlib](https://leanprover-community.github.io/mathlib4_docs/)
  lemmas for formal verification

All tensors derived from the metric are computed lazily and cached.

## Requirements

- Python ≥ 3.12
- [SymPy](https://www.sympy.org/) ≥ 1.14

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

## Installation

```bash
# Clone, then sync the environment
uv sync

# Build a wheel / sdist into dist/
uv build
```

## Quick start

```python
import sympy as sp
from levipy import Manifold

# Coordinates for the 2-sphere
theta, phi = sp.symbols('theta phi', real=True)
r = sp.Symbol('r', positive=True)

# Build a manifold and attach a metric g_{ij}
M = Manifold('S2', coords=[theta, phi])
g = M.metric([[r**2, 0],
              [0, r**2 * sp.sin(theta)**2]])

# Derived geometric quantities
g.christoffel()      # Christoffel symbols  Γ^k_ij
g.riemann()          # Riemann tensor       R^l_ijk
g.ricci_tensor()     # Ricci tensor         R_ij
g.ricci_scalar()     # Ricci scalar         R
g.einstein_tensor()  # Einstein tensor      G_ij
```

Most objects provide a `.display()` method for pretty-printed / LaTeX output.

### Geodesics

```python
geo = g.geodesic_equations()
geo.display()
```

### Parallel transport

```python
from levipy import ParallelTransport

pt = ParallelTransport(g)
pt.equations(curve=[sp.cos(pt.lam), sp.sin(pt.lam)])
```

### Lie bracket

```python
from levipy import lie_bracket

X = [theta, 0]
Y = [0, phi]
lie_bracket(X, Y, coords=[theta, phi])
```

## Lean 4 formal verification bridge

Symbolic computation tells you *what* the curvature of a space is. It does not
tell you the result is *correct* — a typo in a metric or a SymPy simplification
quirk can silently produce a wrong tensor. LeviPy's Lean 4 bridge closes that
gap by translating your computed geometry into **formal statements that the
Lean 4 proof assistant can check against Mathlib**.

This is a *proof-obligation* layer, not an automated prover. Given a fully
computed metric (with its Christoffel symbols, Riemann/Ricci tensors, etc.),
`TheoremBuilder` emits a `.lean` source file containing:

1. The necessary Mathlib imports
2. Coordinate symbols declared as `(x : ℝ)`
3. The metric as a `noncomputable def`
4. One theorem per nonzero Christoffel symbol
5. Riemann flatness / curvature theorems
6. The Ricci scalar theorem
7. Geodesic equation statements

Each theorem comes with a `sorry`-based proof skeleton and **tactic hints that
point at real Mathlib lemmas** — so a human (or Lean's automation) can fill in
the proofs. The output is a set of *checkable* obligations: if the geometry is
wrong, the Lean statements won't close.

Three components make up the bridge:

| Component        | Role                                                        |
| ---------------- | ----------------------------------------------------------- |
| `SymPyToLean4`   | Translates a SymPy expression into a Lean 4 term string     |
| `TheoremBuilder` | Emits a full `.lean` file of proof obligations from a metric |
| `MathlibLinker`  | Suggests relevant Mathlib 4 lemmas/tactics for an expression |

```python
import sympy as sp
from levipy import Manifold, TheoremBuilder

# Name the symbols in Greek so the generated Lean uses θ, φ identifiers
theta, phi = sp.symbols('θ φ', real=True)
M = Manifold('S2', coords=[theta, phi])
g = M.metric([[1, 0], [0, sp.sin(theta)**2]])

builder = TheoremBuilder(
    g,
    manifold_name='Sphere2',
    extra_hypotheses=['(hth : 0 < θ)', '(hth2 : θ < π)'],  # must match the symbol names
)
builder.save('Sphere2.lean')   # writes a Lean 4 source file
```

> **Note for `pip install` users:** the PyPI package ships the Python bridge
> (which *generates* `.lean` files), but not the companion Lean project itself.
> To build and check the generated proofs, clone the
> [GitHub repository](https://github.com/mirajcs/levipy) — the Lean Lake
> project lives in [`lean/`](https://github.com/mirajcs/levipy/tree/main/lean)
> (pinned to a Mathlib toolchain).

Lower-level pieces are also available:

```python
from levipy import SymPyToLean4, MathlibLinker

SymPyToLean4().translate(sp.sin(theta)**2)   # SymPy expr → Lean 4 term string
MathlibLinker()                              # map expressions to Mathlib lemmas
```

## Package layout

```
levipy/
├── geometry/      Manifold, covariant derivative, Lie bracket
├── tensors/       Metric, Christoffel, Riemann, Ricci, Einstein
├── gr/            General-relativity tools (geodesics, parallel transport)
└── lean4/         Lean 4 bridge (translator, theorem builder, Mathlib linker)

lean/              Companion Lean 4 / Lake project (Mathlib-backed proofs)
```

## Development

```bash
# Run the test suite (pytest is a dev dependency)
uv run pytest

# Verbose
uv run pytest -v
```

## License

Released under the [MIT License](LICENSE). © 2026 Miraj Samarakkody.
