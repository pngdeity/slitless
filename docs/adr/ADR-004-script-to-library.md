# ADR-004: Script-to-Library Migration

**Status:** Proposed
**Date:** 2026-05-17

## Context

The codebase has 21 scripts in `python/scripts/`, several of which contain logic that belongs in the library itself. The key anti-pattern is the **duplication** between `comparison_testbed_multi.py` (a 278-line script) and the `comparison_test_multi()` function in `recon.py` (lines 1312-1487 — a 175-line function with the docstring `''' Encapsulated the *comparison_testbed_multi.py* into a function. '''`).

The developer recognized the duplication and already moved the function into the library — but the script still exists. This is typical of the migration pattern: logic starts in a script, gets partially promoted to a library function, but the script is never retired or reduced to a thin wrapper.

Other scripts with significant logic that could (or already partially does) live in the library:

| Script | Lines | Library-like logic |
|---|---|---|
| `comparison_testbed_multi.py` | 278 | Duplicate of `recon.comparison_test_multi()` |
| `final_result_runner.py` | 331 | Solver dispatch, metrics, figure generation |
| `eis_reader_v3.py` | 141 | EIS data pipeline orchestration |
| `generate_dset_v5.py` | 79 | Dataset generation with stats computation |
| `generate_testbed_set.py` | ~100 | Test dataset creation |
| `dataset_analysis.py` | ~200 | Analytics and statistics computation |
| `apj_tomo_plotter.py` | 98 | Tomography demonstration (imports legacy variants) |
| `test_smart.py` | ~50 | Golden test for SMART solver |
| `forward_1d_exp.py` | ~60 | Forward model demonstration |
| `auto_param_searcher.py` | ~100 | Hyperparameter search |

Additionally, scripts like `eis_reader_v3`, `generate_dset_v5`, and `final_result_runner` are explicitly named in the developer familiarity rules as production scripts whose names must not change.

## Decision Options

### Option A: Status Quo
**Description:** All logic in scripts. Library has `comparison_test_multi()` as a library function but `comparison_testbed_multi.py` still exists as a script. Some logic is duplicated.

**Pros:**
- No migration effort.
- Scripts are self-contained — the developer can copy a script and run it standalone.

**Cons:**
- Duplication between script and library function is a maintenance trap — bug fixes must be applied in both places.
- No reusable pipeline functions for composability.
- New experiments require copy-pasting boilerplate from existing scripts.
- Library cannot be used as a library (no entry points for `import slitless; slitless.do_thing()`).

### Option B: Pipeline Functions + Thin Scripts (Recommended)
**Description:** Move core pipeline logic into dedicated library modules. Scripts become thin wrappers (10-30 lines each) that import and call library functions. Old script names and `python scripts/<name>.py` invocation preserved.

**New library modules:**

```python
# slitless/pipeline.py — high-level orchestration functions

def generate_dataset(output_dir, stats_path, lamdim=21, ...):
    """Logic from generate_dset_v5.py"""
    ...

def run_eis_reader(pathdir, savedir, wavelength, ...):
    """Logic from eis_reader_v3.py"""
    ...

def run_final_results(data_file, method, savepath, ...):
    """Logic from final_result_runner.py"""
    ...

def compare_solvers(path_data, data, savepath, solver='scipy', ...):
    """Thin wrapper around recon.comparison_test_multi()"""
    ...
```

**Thin scripts (example):**

```python
# scripts/eis_reader_v3.py  (old name preserved)
from slitless.pipeline import run_eis_reader
from slitless.config import config
run_eis_reader(
    pathdir=str(config.data_root / 'eis_data'),
    savedir=str(config.data_root / 'eis_data/datasets/dset_v4'),
)
```

**Dealing with `comparison_testbed_multi.py`:** Delete the script (it's fully duplicated by `recon.comparison_test_multi()`). Add a thin `scripts/compare_solvers.py` that imports and calls `comparison_test_multi()`.

**Pros:**
- Duplication eliminated: `comparison_test_multi()` is the single implementation.
- Library gains reusable, composable pipeline functions.
- Scripts become trivially short and easy to understand.
- **Production script names preserved**: `eis_reader_v3.py`, `generate_dset_v5.py`, `final_result_runner.py` still exist at the same paths.
- Developer still types `python scripts/eis_reader_v3.py` — workflow unchanged.
- Pipeline functions can be called from Jupyter notebooks or ad-hoc Python for exploration.

**Cons:**
- One-time migration: extract logic from 3-5 key scripts into `pipeline.py` (~200-400 lines new code, ~500 lines deleted from scripts).
- Scripts that use `if __name__ == "__main__"` blocks currently set up matplotlib backends, parse command-line args, etc. — thin wrappers must preserve this setup code.
- Risk of breaking subtle behavior (e.g., script sets `matplotlib.use('Agg')` before imports — thin wrapper must do the same).

### Option C: Full CLI with argparse
**Description:** Every script becomes a subcommand of `python -m slitless.cli`. For example: `python -m slitless.cli eis-reader --output dset_v4`.

**Pros:**
- Standardized interface — all tools accessible from one entry point.
- argparse provides `--help`, type validation, and consistent UX.

**Cons:**
- **Violates developer familiarity rule 2**: production script names change. The developer types `python scripts/eis_reader_v3.py` today; forcing `python -m slitless.cli eis-reader` is ongoing friction.
- All 21 scripts need CLI wrappers written.
- Argparse is rigid — the current pattern of editing a script's hardcoded variables and re-running is faster for a researcher iterating on parameters than typing CLI flags every time.
- Over-engineered for a single-developer exploratory research codebase.

### Option D: Clean Up Duplication Only
**Description:** Delete `comparison_testbed_multi.py` (fully duplicated by `comparison_test_multi()`). Everything else stays in scripts. No new library modules.

**Pros:**
- Solves the worst duplication with minimal change.
- No migration risk.

**Cons:**
- Other scripts still contain library-worthy logic.
- Does not create reusable pipeline functions.
- The library's `recon.py` still has a script-duplicated function living in it — not clean separation.

## Developer Familiarity Impact

| Question | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| `python scripts/eis_reader_v3.py` still works? | Yes | Yes (same name, thinner) | No (must use CLI) | Yes |
| `python scripts/comparison_testbed_multi.py` still works? | Yes | No (retired; logic in `comparison_test_multi()`) | No | No (deleted) |
| Developer edits script to change params? | Yes | Yes (edits thin wrapper) | Must use CLI flags | Yes |
| `from slitless.pipeline import generate_dataset` possible? | No | Yes | Yes | No |
| Duplication between script and library? | Yes | No | No | Partially (only worst case fixed) |
| Migration cost | None | Medium (~500 lines moved) | High (rewrite 21 scripts) | Low (delete 1 script) |

## Recommendation

**Option B** with a phased rollout:

1. **Phase 1 (immediate):** Delete `comparison_testbed_multi.py`. The `comparison_test_multi()` function in `recon.py` is the canonical implementation. This eliminates the worst duplication with zero new code.

2. **Phase 2 (next sprint):** Create `slitless/pipeline.py` with `generate_dataset()`, `run_final_results()`, and `run_eis_reader()`. Convert the 3 corresponding scripts to thin wrappers. Preserve script names exactly.

3. **Phase 3 (later):** Move `test_smart.py` into a proper test directory (see ADR-007). Move `apj_tomo_plotter.py` to use canonical forward ops (see ADR-006).

This phased approach respects the developer familiarity constraints: each phase is individually reversible, no production script name changes until the library equivalent is proven usable, and the developer can opt out of any phase.
