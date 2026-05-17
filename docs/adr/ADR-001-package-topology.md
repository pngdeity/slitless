# ADR-001: Package Topology

**Status:** Proposed
**Date:** 2026-05-17

## Context

The slitless package currently has a single flat structure: all 11 modules live directly under `slitless/` (plus one subdirectory `networks/` for the U-Net architecture). The `__init__.py` exports nothing except `__version__`. All consumer code uses explicit submodule imports (`from slitless.recon import scipy_solver`).

The flat layout creates three problems:
1. **No discoverability signal** — a new contributor cannot tell from `import slitless` what the public modules are, nor which ones depend on which.
2. **`recon.py` is 1487 lines** with 13 solver functions, 3 classes, and a script-duplicating function (`comparison_test_multi` at line 1312). It is unmanageably large but cannot be split without breaking imports.
3. **No boundaries** — the ML pipeline (U-Net, training, evaluation) is interleaved with physics (forward model, EIS tools) and solvers, making responsibility unclear.

The sole developer's workflow is: edit a library file → run a script in `python/scripts/` → view matplotlib output. Any topology change must not disrupt this.

## Developer Familiarity Constraint

Per the project's developer familiarity rules:
1. `from slitless.recon import scipy_solver` must continue to work.
2. `from slitless.forward import Source, Imager` must continue to work.
3. Daily workflow of editing-library-and-running-scripts must not change.
4. No ongoing friction without a corresponding productivity gain.
5. Any new convention must be optional.

## Decision Options

### Option A: Status Quo
**Description:** Single flat package. All modules at `slitless/` level with `networks/` subdirectory. `__init__.py` exports only `__version__`.

**Pros:**
- Zero migration cost.
- All current imports work unchanged.
- Developer's muscle memory intact.

**Cons:**
- `recon.py` cannot be split without breaking imports.
- No logical grouping — solvers, ML, and forward model are indistinguishable.
- Onboarding friction for new contributors.

### Option B: Three Subpackages with Facade (Recommended)
**Description:** Restructure into three subpackages:
- `slitless/core/` — `forward.py`, `config.py`, `data_loader.py`, `common.py`
- `slitless/solvers/` — solver functions extracted from `recon.py` into one file per solver (e.g., `scipy.py`, `smart.py`, `tomoinv.py`, `nn.py`, `diffusion.py`)
- `slitless/ml/` — `unet.py` (moved from `networks/`), `train.py`, `evaluate.py`, `measure.py`

Keep `slitless/recon.py` as a thin facade that re-exports everything from the subpackages:

```python
# slitless/recon.py — facade, preserves all old imports
from slitless.solvers.scipy import scipy_solver, scipy_solver_parallel, scipy_solver_parallel2
from slitless.solvers.smart import smart, smart2, smart2_twostage
from slitless.solvers.tomoinv import tomoinv, tomoinv0
from slitless.solvers.gradient import grad_descent_solver, gauss_pmf_fitter, gauss_pmf_fitter2
from slitless.solvers.nn import nn_solver
from slitless.solvers.diffusion import diffusion_solver
from slitless.core.common import Reconstructor, Reconstructor_Multi, Recon
from slitless.core.comparison import comparison_test_multi
```

`__init__.py` remains minimal (just `__version__`).

**Pros:**
- All old imports preserved: `from slitless.recon import scipy_solver`, `from slitless.forward import Source, Imager` both work unchanged.
- `recon.py` shrinks from 1487 lines to ~30 lines of re-exports.
- Each solver becomes a manageable file (100-300 lines).
- Logical grouping makes the package navigable.
- New convention is optional — old flat imports keep working.
- Zero daily workflow change: edit library file → run script → view plot.

**Cons:**
- ~1200 lines of `recon.py` must be split into multiple files (one-time effort, ~1 hour).
- Split risks subtle import-order issues (solvers internally import forward, etc.).
- Must ensure `setup.py` lists all subpackages.

### Option C: Full Namespace Split with Typed Public API
**Description:** Three subpackages as in Option B, plus a proper `__init__.py` with typed re-exports. Consumers would use `from slitless import scipy_solver` or dotted access. Old submodule imports would emit deprecation warnings.

**Pros:**
- Cleanest public API surface.
- Enables `mypy` typing of the public interface.

**Cons:**
- **Violates developer familiarity rule 1**: `from slitless.recon import scipy_solver` would get a deprecation warning on every run — ongoing friction.
- Requires rewriting all script imports.
- Over-engineered for a codebase with one developer.

### Option D: Two Packages (`slitless` + `scripts/`)
**Description:** Move `python/scripts/` outside the package entirely (e.g., to repo root `scripts/` or into its own non-package directory). Library remains flat but scripts are clearly separated.

**Pros:**
- Clean separation between library and driver code.
- Scripts can import via relative paths without `slitless.` prefix issues.

**Cons:**
- Scripts already work fine with `from slitless.X import Y`; moving them changes `sys.path` behavior.
- The 21 scripts cross-import each other (e.g., scripts call functions defined in other scripts); moving them breaks these relationships.
- Violates developer familiarity rule 2 (script names can't change, but their import paths would).
- Does not address the `recon.py` monolith problem.

## Developer Familiarity Impact

| Question | Option A | Option B (Recommended) | Option C | Option D |
|---|---|---|---|---|
| Daily workflow changes? | None | None | Must use new import paths | Script paths change |
| `from slitless.recon import scipy_solver` preserved? | Yes | Yes | Deprecation warning | Yes |
| `from slitless.forward import Source` preserved? | Yes | Yes (`forward.py` stays at `slitless/core/` but importable via `slitless.forward` if a facade is kept, or must add re-export) | Deprecation warning | Yes |
| New conventions? | None | Split solvers by file | Typed public API | Script location convention |
| Migration cost | None | One-time split of `recon.py` | High — rewrite all imports | Medium — update 21 script paths |

**Note on `forward.py` imports:** To preserve `from slitless.forward import Source`, move `forward.py` to `slitless/core/forward.py` and add a thin `slitless/forward.py` facade that re-exports everything. Alternatively, leave `forward.py` at the top level and only move solvers.

## Recommendation

**Option B** with the following refinement: extract only `recon.py` content into `slitless/solvers/`; leave all other modules (`forward.py`, `config.py`, `data_loader.py`, `eistools.py`, `train.py`, `evaluate.py`, `measure.py`, `plotting.py`, `common.py`) at the `slitless/` level. This minimizes the change surface — only `recon.py` gets the facade treatment. The `networks/unet.py` stays where it is (it's one file, not worth moving).

The `slitless/recon.py` facade would be the only new file pattern. All other imports work unchanged. This is the lowest-risk approach that still solves the monolith problem.
