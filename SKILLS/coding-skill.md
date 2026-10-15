---
name: aarsh-coding-style
description: >
  Aarsh's personal coding standard for Python projects — especially physics,
  simulation, CAD, and robotics code. Use this skill whenever Aarsh asks you
  to write, extend, debug, or review any Python code. Covers file structure,
  docstring conventions, stdlib-first policy, physics variable naming, README
  updates, and a pre-delivery bug-check checklist. Trigger on any code
  request, including "write a script", "add a feature", "debug this", or
  "refactor".
---

# Aarsh's Python Coding Standard

This skill governs every Python file you write or edit for Aarsh.
Read the whole thing before touching a single line of code.

---

## 1. Stdlib-First Rule

**Default to the Python standard library. Only reach for a third-party
package when the stdlib genuinely cannot do the job.**

| Task | Use stdlib | Reach for third-party only if |
|---|---|---|
| File I/O | `pathlib`, `os`, `shutil` | — |
| Math / linear algebra | `math`, `cmath`, `fractions` | matrix ops → `numpy` |
| Data structures | `collections`, `heapq`, `bisect` | — |
| Argument parsing | `argparse` | — |
| Logging | `logging` | — |
| JSON / CSV / INI | `json`, `csv`, `configparser` | — |
| HTTP | `urllib.request` | complex REST → `requests` |
| Concurrency | `threading`, `multiprocessing`, `concurrent.futures` | — |
| Time / profiling | `time`, `timeit`, `cProfile` | — |
| Unit tests | `unittest` | pytest only if fixtures get complex |
| Numerics / simulation | — | `numpy`, `scipy` |
| Plotting | — | `matplotlib` |
| CAD / geometry | — | `OCC` (pythonocc-core) |
| ML / PINNs | — | `torch` or `jax` |

**If you're about to write a helper that reimplements something already in
`itertools`, `functools`, `collections`, or `math`, stop and use the stdlib
version instead.**

---

## 2. Every File Gets a Header and a Footer

### 2.1 Header (top of file, before any imports)

```python
"""
================================================================================
FILE: <filename>.py
================================================================================
PURPOSE:
    One paragraph. What problem does this file solve? Why does it exist?
    What is the single responsibility of this module?

INPUTS:
    - List every external input: CLI args, file paths, environment variables,
      function arguments if this is a library module.

OUTPUTS:
    - List every output: files written, return values, printed results,
      side effects.

DEPENDENCIES:
    stdlib  : list, of, used, stdlib, modules
    third-party : list (e.g. numpy>=1.24, scipy)
    internal    : list (e.g. utils.geometry, models.pinn)

PHYSICS / DOMAIN CONTEXT (include if applicable):
    - What physical system or CAD domain does this code model?
    - What assumptions are baked in (e.g. incompressible flow, linear elastic)?
    - What SI units are used unless otherwise noted?

AUTHOR: Aarsh Jain
LAST MODIFIED: <YYYY-MM-DD>
================================================================================
"""
```

### 2.2 Footer (bottom of file, after all code)

```python
# ==============================================================================
# END OF FILE: <filename>.py
# ==============================================================================
# SUMMARY:
#   One-liner recap of what this file does.
#
# USAGE EXAMPLE:
#   $ python <filename>.py --arg value
#   or
#   from <module> import <SomeClass>
#   obj = SomeClass(...)
#   result = obj.run()
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Any edge cases not yet handled
#   - [ ] Performance notes if relevant
#
# CHANGELOG:
#   YYYY-MM-DD  Initial implementation
#   YYYY-MM-DD  Added <feature>
# ==============================================================================
```

Both header and footer are **mandatory** on every `.py` file, no exceptions —
library modules, scripts, test files, CLI entrypoints, all of them.

---

## 3. Code Structure Inside a File

Follow this top-to-bottom order within every file:

```
[HEADER DOCSTRING]          ← section 2.1

[STDLIB IMPORTS]            ← alphabetical
[THIRD-PARTY IMPORTS]       ← alphabetical, after a blank line
[INTERNAL IMPORTS]          ← alphabetical, after a blank line

[PATH VARIABLES]            ← relative paths only — see Section 3.1
                              e.g.  DATA_DIR = Path("data/raw")

[CONSTANTS]                 ← ALL_CAPS, with units in comments
                              e.g.  G = 9.81  # m/s², gravitational acceleration

[TYPE ALIASES / PROTOCOLS]  ← if needed

[HELPER / UTILITY FUNCTIONS]   ← small, single-purpose, no side effects

[CORE CLASSES / MAIN FUNCTIONS]

[CLI ENTRYPOINT]            ← if script
    if __name__ == "__main__":
        ...

[FOOTER COMMENT]            ← section 2.2
```

### 3.1 Path Variables — Local References Only

**All file paths must be relative to the project root. Never hardcode an
absolute or global path.**

```python
from pathlib import Path

# ── PATH VARIABLES (declare before CONSTANTS) ─────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent        # project root
DATA_DIR    = ROOT_DIR / "data" / "raw"              # input data
OUTPUT_DIR  = ROOT_DIR / "output"                    # results / artifacts
CONFIG_DIR  = ROOT_DIR / "config"                    # config files
LOG_DIR     = ROOT_DIR / "logs"                      # log files
```

**Rules:**
- Always derive paths from `Path(__file__).resolve().parent` — never
  from `os.getcwd()` or a hardcoded string like `"/home/aarsh/project/..."`.
- `pathlib.Path` only — no bare string concatenation for paths.
- Path variables go in their own `PATH VARIABLES` block, **above** `CONSTANTS`.
- Any function that reads or writes a file takes a `Path` argument — it does
  not construct the path internally from a hardcoded root.

```python
# ✅ Correct — caller supplies the path
def load_mesh(mesh_path: Path) -> np.ndarray:
    data = np.load(mesh_path)
    ...

# ❌ Wrong — path baked in
def load_mesh() -> np.ndarray:
    data = np.load("/home/aarsh/project/data/mesh.npy")
    ...
```

- If a path needs to be user-configurable, expose it via `argparse` or a
  config file, defaulting to the `ROOT_DIR`-relative value:

```python
parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
```

---

### Where to Add New Features

- New **utility function** → add inside the `HELPER / UTILITY FUNCTIONS`
  block, in alphabetical order by function name.
- New **class or major function** → add inside `CORE CLASSES / MAIN
  FUNCTIONS`, after existing ones (append, don't scatter).
- New **constant** → add to `CONSTANTS` block immediately; include value,
  type, and unit.
- Never insert code between unrelated existing functions — find the right
  block.

---

## 4. Docstrings and Inline Comments

### Function / Method Docstring

```python
def solve_navier_stokes_1d(
    rho: float,
    mu: float,
    u_initial: np.ndarray,
    x: np.ndarray,
    dt: float,
    t_end: float,
) -> np.ndarray:
    """
    Solve 1D Navier-Stokes using explicit Euler time-stepping.

    Parameters
    ----------
    rho : float
        Fluid density [kg/m³].
    mu : float
        Dynamic viscosity [Pa·s].
    u_initial : np.ndarray, shape (N,)
        Initial velocity field [m/s].
    x : np.ndarray, shape (N,)
        Spatial grid [m].
    dt : float
        Time step [s]. Must satisfy CFL: dt < dx²·rho / (2·mu).
    t_end : float
        Simulation end time [s].

    Returns
    -------
    np.ndarray, shape (N,)
        Velocity field at t=t_end [m/s].

    Raises
    ------
    ValueError
        If CFL condition is violated.
    """
```

**Rules:**
- Always include units in brackets `[unit]` for every physical quantity.
- State the array shape in the type annotation comment.
- Call out any constraint or precondition (CFL, positive-definite matrix, etc.).
- No one-liner docstrings for physics functions — they always get the full
  NumPy-style block.

### Inline Comments

- Use inline comments for **non-obvious** lines only; don't narrate obvious code.
- Physics formula lines **always** get a comment with the equation reference:

```python
# Navier-Stokes momentum: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u
du_dt = -(u * du_dx) - (1/rho) * dp_dx + (mu/rho) * d2u_dx2
```

---

## 5. Physics and Engineering Conventions

### Variable Naming

Use conventional physics symbols as variable names — don't rename them to
vague names like `val` or `param`.

| Quantity | Variable name |
|---|---|
| Density | `rho` |
| Dynamic viscosity | `mu` |
| Kinematic viscosity | `nu` |
| Pressure | `p` |
| Velocity (scalar / vector) | `u`, `v`, `w` or `vel` |
| Temperature | `T` |
| Stress tensor | `sigma` |
| Strain tensor | `epsilon` |
| Force | `F` |
| Torque / Moment | `tau` |
| Young's modulus | `E` |
| Poisson's ratio | `nu_p` (distinguish from kinematic viscosity) |
| Reynolds number | `Re` |
| Mach number | `Ma` |
| Time / time step | `t`, `dt` |
| Spatial coords | `x`, `y`, `z` |
| Grid spacing | `dx`, `dy`, `dz` |
| Angular quantities | `omega`, `theta`, `phi` |

### Unit System

- **Default: SI throughout.** State deviations explicitly.
- Every constant and every function parameter that carries units must have
  its unit in a comment or docstring.
- If a function accepts inputs in non-SI units, convert at the top of the
  function body and note it:
  ```python
  # Convert input from degrees to radians (SI)
  theta_rad = math.radians(theta_deg)
  ```

### Numerical Safety

- Never compute differences of large nearly-equal floats without noting the
  cancellation risk.
- Guard against division by zero for any denominator derived from user input.
- Use `np.linalg.cond()` or log a warning if inverting matrices whose
  conditioning could be poor.

---

## 6. Error Handling

```python
# Prefer explicit ValueError / TypeError over bare Exception
if dt > cfl_limit:
    raise ValueError(
        f"CFL violated: dt={dt:.4e} exceeds limit {cfl_limit:.4e}. "
        f"Reduce time step or increase grid spacing."
    )

# Use logging, not print()
import logging
logger = logging.getLogger(__name__)
logger.info("Solver initialized with %d nodes, dt=%.3e s", n_nodes, dt)
logger.warning("Condition number %.2e — solution may be inaccurate", cond)
```

- **Never** use bare `except:` or `except Exception:` without re-raising or
  logging the full traceback.
- Physics code should fail loudly with a descriptive message — silent
  wrong answers are worse than crashes.

---

## 7. Type Hints

Use type hints on every function signature. Physics arrays use `np.ndarray`;
be explicit about shape in the docstring.

```python
from __future__ import annotations
from typing import Optional
import numpy as np

def assemble_stiffness_matrix(
    nodes: np.ndarray,          # shape (N, 3)
    elements: np.ndarray,       # shape (M, 4) — tetrahedral
    E: float,
    nu_p: float,
) -> np.ndarray:                # shape (3N, 3N)
```

---

## 8. README Update Rule

**Every code session that adds, changes, or removes functionality must end
with a README.md update.**

The README must always contain:

```markdown
## What This Does
<one paragraph>

## Setup
<pip install / conda env, if applicable>

## Usage
<minimal working example — copy-pasteable>

## File Map
| File | Purpose |
|---|---|
| `solver.py` | Core PDE solver |
| `mesh.py`   | Mesh generation utilities |
| ...         | ... |

## Physics / Domain Assumptions
<bullet list of key model assumptions>

## Changelog
| Date | Change |
|---|---|
| YYYY-MM-DD | Initial commit |
| YYYY-MM-DD | Added feature X |
```

When updating the README:
1. Update the relevant section(s) only — don't rewrite what didn't change.
2. Add a changelog row with today's date and a one-liner.
3. Keep the File Map in sync — add/remove rows when files are added/removed.

---

## 9. Pre-Delivery Bug-Check Checklist

Before handing any code back to Aarsh, run through this list mentally:

```
CODE CORRECTNESS
[ ] Every new function has a test call or usage example (even if informal)
[ ] Array indexing: off-by-one errors checked, especially in FD/FEM loops
[ ] Loop bounds correct — no silent out-of-range slicing
[ ] All branches reachable; no dead `else` clauses
[ ] No mutable default arguments (def f(x, data=[]) is a Python footgun)

PHYSICS
[ ] Units consistent throughout the function/module
[ ] CFL / stability conditions checked where relevant
[ ] Boundary conditions applied in the right place (not off by one node)
[ ] Division by any physics quantity that could be zero is guarded

IMPORTS & DEPENDENCIES
[ ] No import is unused
[ ] Stdlib alternatives exist for every third-party import used
[ ] __future__ annotations imported if using pipe-style type hints

STYLE
[ ] Header docstring present and filled in
[ ] Footer comment present and filled in
[ ] README updated (if any functionality changed)
[ ] No print() left in library code (use logging)
[ ] No hardcoded absolute paths — all paths relative via Path(__file__).parent
[ ] PATH VARIABLES block declared above CONSTANTS block
[ ] Every file-reading function accepts a Path argument, not a hardcoded string

FINAL
[ ] The code runs without error on a minimal smoke-test input
[ ] No TODO left in code without a comment explaining why it's deferred
```

If any box is unchecked, fix it before delivering.

---

## 10. Quick Reference Card

```
paths → relative only, Path(__file__).parent, declared before constants
stdlib first → third-party only when necessary
header docstring → every file, always
footer comment → every file, always
new function → correct block, alphabetical
new constant → CONSTANTS block with unit comment
physics vars → rho, mu, nu, sigma, epsilon, F, T ...
units → SI by default, stated explicitly everywhere
errors → ValueError/TypeError, descriptive message, logging not print
README → update every session, add changelog row
bug check → run the checklist before delivery
```

---

*This skill applies to every Python file written in any Aarsh project —
simulation, CAD pipeline, robotics, PINN training, CLI tools, everything.*
