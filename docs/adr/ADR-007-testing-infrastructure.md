# ADR-007: Testing and Quality Infrastructure

**Status:** Proposed
**Date:** 2026-05-17

## Context

The slitless codebase has **no automated testing, no linting, and no type checking**. The sole test is a script (`test_smart.py` in `python/scripts/`) that the developer runs manually.

The developer's quality workflow is implicit: run a script, view the matplotlib output, and visually confirm it looks right. This works for a single developer but has known failure modes:

1. **Silent numerical drift**: Changes to `forward_op` could alter output by subtle amounts that aren't visible in a plot but accumulate over pipeline runs.
2. **Import rot**: A refactored import path can break all downstream scripts, discovered only when the developer next runs them.
3. **Solver regression**: A performance optimization in `scipy_solver` could break `smart` by changing a shared utility, undetected.
4. **No safety net for refactoring**: Any of the ADRs 001-006 involve moving or wrapping functions — without tests, there's no way to confirm correctness.

The `test_smart.py` script (in `scripts/`) is effectively a "golden test" — it runs a known input through the SMART solver and likely checks output shapes or visualizes results. It can serve as the seed for a proper test suite.

**Codebase metrics relevant to testing:**

| Item | Count |
|---|---|
| Public solver functions | 13 |
| Public forward operators | 8 |
| Dataset classes | 2 |
| Loss/metric functions | 5 |
| Config fields | 23 |
| Lines of library code | ~4500 |

## Decision Options

### Option A: Status Quo
**Description:** No automated tests, no linting, no type checking. Quality via manual script execution.

**Pros:**
- Zero setup cost.
- Developer's existing workflow unchanged.

**Cons:**
- Any refactoring (including ADRs 001-006) cannot be validated automatically.
- Silent regressions are undetectable until visual inspection.
- No guardrails for new contributors.
- Cannot run CI if the project is eventually open-sourced.

### Option B: Minimal pytest + ruff + mypy (Recommended)
**Description:** Add three standard tools with minimal configuration, starting from the existing golden test.

#### pytest
- Add `pytest` to `install_requires` (or a `[dev]` extra).
- Create `python/tests/` directory.
- **Smoke test for all 13 solvers**: Create a tiny `Source` (e.g., 8x8 pixels), instantiate an `Imager`, call each solver, assert output has correct shape and no NaN values.
- **Golden test for `forward_op`**: Compare output on a fixed input against a precomputed numpy array (saved as `tests/data/forward_op_golden.npy`). This catches numerical drift.
- **Import smoke test**: Verify all public modules (`forward`, `recon`, `config`, `data_loader`, `eistools`, `evaluate`, `train`, `measure`, `plotting`, `common`) are importable.
- **Config test**: Verify `SlitlessConfig` instantiates with defaults and all properties return `Path` objects.

**Total tests:** ~20-30, quick to run (<30 seconds).

#### ruff (linter)
- Add `ruff` to dev dependencies.
- Create `pyproject.toml` section with:
  - `line-length = 100` (matches existing style)
  - `indent-width = 4` (matches existing style)
  - Select rules: `E` (pycodestyle errors), `F` (pyflakes), `W` (pycodestyle warnings) — the minimal set that catches real bugs.
  - **Exclude**: `E501` (line length — many long lines in existing code), docstring rules (no docstring requirements initially).
  - Per-file ignores: `__init__.py` (imports unused), `scripts/*` (not library code).
- Run `ruff check .` — accept that some warnings exist and fix incrementally.

#### mypy (type checker)
- Add `mypy` to dev dependencies.
- Configure `pyproject.toml`:
  - `ignore_missing_imports = true` (many dependencies lack stubs: `eispac`, `skimage`, `pqdm`, `denoising-diffusion-pytorch`)
  - `disallow_untyped_defs = false` (existing code is untyped)
  - `warn_return_any = false` (permissive mode)
- Start with `mypy slitless/config.py` (already typed via dataclass) and expand file by file.

**Pros:**
- The smoke test for all solvers catches shape errors and NaN propagation with ~0.5s runtime.
- The golden test for `forward_op` catches the most critical regression: forward model numerical drift.
- `ruff` catches real bugs (undefined names, unused imports hiding typos) without requiring mass reformatting.
- `mypy` can be adopted incrementally — start with the already-typed `config.py`.
- Tools are optional — developer can keep running scripts as before.
- `pyproject.toml` is a standard Python project convention, not a disruption.

**Cons:**
- `ruff check .` will produce warnings on first run (undefined names in scripts, long lines). This is expected — fix incrementally, not all at once.
- `mypy` on untyped code produces many errors in permissive mode. The `disallow_untyped_defs = false` setting suppresses the worst.
- The golden test requires committing a binary `.npy` file to the repo (~few KB). This is standard practice for numerical libraries.
- Some solvers (diffusion, NN) require GPU or large model downloads — their smoke tests must be skipped if CUDA/models are unavailable.

### Option C: Full Test Suite + CI + Coverage Gates
**Description:** Comprehensive test suite (>100 tests), GitHub Actions CI on push, code coverage enforced at 80%+, linting must pass for merge.

**Pros:**
- Production-grade quality infrastructure.
- CI catches regressions automatically.
- Coverage gates ensure thorough testing.

**Cons:**
- **Massive over-investment** for a single-developer research codebase.
- CI requires GitHub Actions setup, model file hosting, and GPU runners (for NN/diffusion tests).
- Coverage gates would require testing 4500 lines — including matplotlib plotting code and EIS data download functions — which is impractical.
- Violates developer familiarity rule 4: maintaining a full CI pipeline is ongoing friction without a corresponding productivity gain at this stage.

## Developer Familiarity Impact

| Question | Option A | Option B | Option C |
|---|---|---|---|
| Developer must run tests before committing? | No | No (optional) | Yes (CI gate) |
| Developer can still run scripts directly? | Yes | Yes | Yes |
| `ruff` requires reformatting existing code? | No (permissive config) | No | Yes (strict config) |
| New tool install required? | No | `pip install pytest ruff mypy` | Same + CI config |
| Test suite runtime | 0s | <30s | 5-10 min |
| Catches `forward_op` numerical regression? | No | Yes (golden test) | Yes |
| Catches import breakage (e.g., after refactor)? | No | Yes (import smoke test) | Yes |

## Recommendation

**Option B** with the following minimal initial implementation:

### Phase 1: Setup (day 1)
1. Add `pytest`, `ruff`, `mypy` to an `[dev]` extras group in `setup.py`.
2. Create `pyproject.toml` with ruff and mypy configuration.
3. Create `python/tests/` directory.

### Phase 2: Smoke Tests (day 1-2)
4. **Golden test**: Save `forward_op` output of a known input (e.g., a 16x16 `Source` with known intensity/velocity/linewidth) as `tests/data/forward_op_golden.npy`. Test that current `forward_op` produces bitwise-identical output.
5. **Solver smoke tests**: For each of the 13 solvers, run on an 8x8 toy problem, assert:
   - Return value is not None
   - Recon has shape `(3, 8, 8)` or `(N, 3, 8, 8)`
   - No NaN values in recon
   - (Skip NN and diffusion if models unavailable)
6. **Import smoke test**: `import slitless.forward`, `import slitless.recon`, etc.
7. **Config test**: `SlitlessConfig()` instantiates cleanly.

### Phase 3: CI (when needed)
8. When the project is shared with others or the developer wants automated validation, add a GitHub Actions workflow that runs `pytest` and `ruff check` on push.

### ruff configuration (targeting existing style)
```toml
[tool.ruff]
line-length = 100
indent-width = 4

[tool.ruff.lint]
select = ["E", "F", "W"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"scripts/*" = ["F401", "F403", "F821"]
```

### mypy configuration (permissive start)
```toml
[tool.mypy]
ignore_missing_imports = true
disallow_untyped_defs = false
warn_return_any = false
files = ["python/slitless/config.py"]
```

### Key decision: Golden test data location
The golden test data (`forward_op_golden.npy` and solver test data) should live in `tests/data/`. This is a ~10 KB `.npy` file — small enough to commit to git. The golden test generates the expected output once and thereafter detects numerical drift on every run.

### Key decision: Dealing with the NN/Diffusion solvers
The `nn_solver` and `diffusion_solver` require trained models (hundreds of MB) and potentially CUDA. Their smoke tests should:
- Check `torch.cuda.is_available()` and skip if not
- Check for model file existence and skip if absent
- Use `pytest.skip()` with a descriptive message

This follows the pattern: tests that depend on external resources are skipped gracefully, not failed.
