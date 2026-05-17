# ADR-002: Solver Interface Standardization

**Status:** Proposed
**Date:** 2026-05-17

## Context

`recon.py` contains 13 solver functions with no shared interface. Each solver has a different signature and different return conventions. There is no common contract for "what is a solver."

**Current solver signatures (simplified):**

| Solver | Parameters beyond `imager` | Returns |
|---|---|---|
| `smart(imager, n_iters=50, t=1, tol=1e-6, ...)` | 8 kwargs | `(recon_3d_array, loss_array)` |
| `smart2(imager, n_iters=50, ...)` | 10 kwargs | `(recon_3d_array, loss_array)` |
| `smart2_twostage(imager, ...)` | 12 kwargs | `(recon_3d_array, losses_dict)` |
| `scipy_solver(imager, method='L-BFGS-B', ...)` | 5 kwargs | `(recon_3d_array, opt_result_or_loss)` |
| `scipy_solver_parallel(imager, ...)` | 5 kwargs | `(recon_3d_array, loss_array)` |
| `scipy_solver_parallel2(imager, ...)` | 8 kwargs | `(recon_3d_array, loss_array)` |
| `grad_descent_solver(imager, lr=0.1, ...)` | 7 kwargs | `(recon_3d_array, loss_array)` |
| `tomoinv(imager, ...)` | 5 kwargs | `recon_3d_array` (no loss) |
| `tomoinv0(imager, ...)` | 3 kwargs | `recon_3d_array` (no loss) |
| `nn_solver(imager, model_path='', ...)` | 3 kwargs | `(recon_3d_array, 0)` (dummy loss) |
| `diffusion_solver(imager, model_path='', ...)` | 5 kwargs | `(recon_3d_array, norms)` (norms not loss) |
| `prior_solver(imager, ...)` | 4 kwargs | `(recon_3d_array, None)` |
| `gauss_pmf_fitter(line)` | 1 arg (not imager) | `intensity, velocity, linewidth` |
| `gauss_pmf_fitter2(cube, ...)` | 4 args | `(param3d, corr, fit)` |

Because there is no shared interface:
- `Reconstructor.solve()` calls `self.solver(imager=self.imager, **self.solver_params)` and hopes it works (line 60-63 of `recon.py`).
- `Reconstructor_Multi.solve()` dispatches on `solver.__name__` in the manually-maintained if/elif chain at `comparison_test_multi()` (lines 1341-1396).
- Some solvers return 2-tuples, some return 1-tuples, `diffusion_solver` returns `norms` instead of losses.
- Units vary: some solvers work in pixel units, some in physical units with wavelength/velocity conversion baked in.

## Decision Options

### Option A: Status Quo
**Description:** Heterogeneous signatures. Each solver takes different kwargs and returns different things. No shared protocol.

**Pros:**
- Zero migration cost.
- Each solver is free to evolve its own interface.
- Developer already knows each solver's quirks.

**Cons:**
- Adding a new solver requires updating the if/elif dispatch in `comparison_test_multi()` (13 branches and growing).
- Cannot write generic solver-comparison code or metrics computation.
- `Reconstructor`'s `recon_inv_transform` (line 85-94) has a hardcoded `if self.solver.__name__ == 'nn_solver':` branch — fragile string matching.
- Onboarding: a new person must read each solver's docstring individually.

### Option B: Protocol-Based Solver Interface (Recommended)
**Description:** Define a `SolverResult` dataclass and a Protocol that all solvers conform to. Solvers accept `(imager: Imager, truth: Optional[np.ndarray] = None, **solver_params) -> SolverResult`.

```python
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
import numpy as np

@dataclass
class SolverResult:
    recon: np.ndarray          # shape (3, H, W) in pixel units
    losses: np.ndarray         # loss per iteration, shape (n_iters,)
    metadata: dict = field(default_factory=dict)  # optional: timing, solver-specific info

@runtime_checkable
class Solver(Protocol):
    def __call__(self, imager: Imager, truth: Optional[np.ndarray] = None, **kwargs) -> SolverResult: ...
```

Dispatchers (`Reconstructor`, `Reconstructor_Multi`, `comparison_test_multi`) now unpack `result.recon`, `result.losses` uniformly.

**Pros:**
- `Reconstructor.solve()` becomes generic — no `solver.__name__` string matching.
- `comparison_test_multi()` dispatch drops from 60 lines to a dict lookup.
- Metrics computation (RMSE, SSIM, bias) can be done uniformly on `result.recon`.
- Each solver can still accept arbitrary kwargs via `**solver_params`.
- Protocol is structural (not inheritance) — no ABC registration needed.

**Cons:**
- 13 solvers need their return types wrapped (one-time, ~5 minutes each).
- `diffusion_solver` currently returns `norms` (gradient norms, not losses) — must decide whether to put in `losses` or `metadata`.
- `tomoinv` / `tomoinv0` currently don't return losses — must return an empty array or None in `losses` and handle downstream.
- `gauss_pmf_fitter` / `gauss_pmf_fitter2` don't take `Imager` — may be excluded from the protocol.

### Option C: ABC-Based Solver Registry
**Description:** An abstract base class with a formal plugin registry. Each solver subclasses `SolverBase` and registers via a decorator. The registry provides name-based lookup.

**Pros:**
- Strongly typed — `mypy` can check conformity at analysis time.
- Self-documenting: `Solver.registry` lists all available solvers.
- Plugin discovery without string matching.

**Cons:**
- **Requires rewriting all 13 solver functions as classes** — massive disruption.
- Inheritance couples solvers to framework code they currently don't depend on.
- Violates developer familiarity rule: existing function-based solver calls must change.
- Over-engineered for 13 solvers in a single-developer codebase.

### Option D: Function Registry Dict
**Description:** A simple dictionary: `SOLVERS = {"scipy": scipy_solver, "smart": smart, "nn": nn_solver, ...}`. Solver functions keep their heterogeneous signatures exactly as-is. Dispatchers look up by key. No return type changes.

**Pros:**
- **Least disruptive** — zero changes to solver function signatures or bodies.
- Eliminates the if/elif chain in `comparison_test_multi()` (replaced by `SOLVERS[solver_name]`).
- Adds exactly one new convention: a central registry dict.
- Can be adopted incrementally — add entries one at a time.

**Cons:**
- Still no shared return type — `Reconstructor` must still check `solver.__name__` for the NN special case.
- Adding a solver still requires knowing the implicit conventions (must return `(recon, loss)`).
- Does not solve the heterogeneous return problem; only solves dispatch.

## Developer Familiarity Impact

| Question | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| Existing solver signatures changed? | No | Yes (return type wrapped) | Yes (rewrite as classes) | No |
| `Reconstructor(solver=scipy_solver)` breaks? | No | No (Protocol is structural) | Yes (must pass instance) | No |
| Script calls like `smart(imager=Imgr)` break? | No | Slight change (unwrap `.recon`) | Yes (API changes) | No |
| `comparison_test_multi()` dispatch simplified? | No | Yes (`.recon`, `.losses` uniform) | Yes (registry) | Yes (dict lookup) |
| Migration cost | None | One-time return wrapping | High — class rewrite | One-time dict creation |

## Recommendation

**Option D** for immediate adoption, with a clear path to Option B later.

Immediate step: create `slitless/solver_registry.py` containing a `SOLVERS` dict. Update `comparison_test_multi()` to use it. This eliminates the 60-line if/elif chain with zero changes to solver functions.

Future step: once the registry is stable, introduce `SolverResult` and migrate solvers one at a time. The dataclass `recon` field establishes the convention that recon is always in **pixel units** (matching `Source(pix=True).param3d`), resolving the unit ambiguity.

The `gauss_pmf_fitter` and `gauss_pmf_fitter2` functions should be excluded from the solver protocol — they are fitting utilities, not reconstruction solvers. They don't take `Imager` and conceptually belong in a `fitting.py` utility module, not the solver system.
