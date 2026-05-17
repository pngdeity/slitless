# ADR-006: Forward Model Variant Strategy

**Status:** Proposed
**Date:** 2026-05-17

## Context

`forward.py` (740 lines, the second-largest module) contains 4 "variant" functions that are explicitly marked as deprecated but still imported by production code:

| Function | Lines | Deprecated comment? | Still imported by |
|---|---|---|---|
| `forward_op_tomo_3d_k3` | 137-155 | Yes (line 136) | `apj_tomo_plotter.py` (line 3) |
| `forward_op_tomo_3d_v0` | 160-199 | Yes (line 159) | `apj_tomo_plotter.py` (line 3) |
| `forward_op_tomo_3d_transpose_k3` | 256-278 | Yes (line 255) | `apj_tomo_plotter.py` (line 4) |
| `forward_op_tomo_2d_transpose` | 239-253 | No comment, but 2D-only | No known callers |

The canonical versions are:

| Function | Lines | Description |
|---|---|---|
| `forward_op_tomo_3d` | 201+ | 3D forward projection with arbitrary spectral orders |
| `forward_op_tomo_3d_transpose` | 280-318 | 3D back-projection with arbitrary spectral orders |

The `_k3` variants hardcode 3 orders `[0, -1, 1]`. The `_v0` variant adds ±2nd order support but has a different internal structure. The canonical versions generalize both with the `orders=[0,-1,1]` parameter.

**Lines consumed by variants:** ~140 lines (including `forward_op_tomo_2d` and `forward_op_tomo_2d_transpose` which are 2D-only and appear to have no callers).

The only script importing the deprecated variants is `apj_tomo_plotter.py` (98 lines), which imports both `_k3` and `_v0` variants alongside the canonical versions and appears to compare their outputs. It's a diagnostic/validation script, not a production pipeline.

## Developer Familiarity Constraint

Per the developer familiarity rules, `apj_tomo_plotter.py` is not listed as a production script (the production scripts are `eis_reader_v3`, `generate_dset_v5`, `final_result_runner`). This means it can be modified, but the developer must still be able to run it if he wants to.

## Decision Options

### Option A: Status Quo
**Description:** Keep deprecated variants in `forward.py` indefinitely. The comment "still imported by scripts/" acknowledges they're needed but defers removal.

**Pros:**
- Zero risk of breaking anything.
- `apj_tomo_plotter.py` works as-is.

**Cons:**
- 140 lines of dead/deprecated code in the second-largest module, making `forward.py` harder to read and maintain.
- New contributors will read the deprecated functions and wonder which one to use.
- Every edit to `forward.py` must scroll past the deprecated variants.
- The `import` comment ("DEPRECATED: use X instead") is a lie — it says "use X" but X can't be used because the deprecated variant hasn't been removed.

### Option B: Legacy Module with Deprecation Warnings (Recommended)
**Description:** Move the 4 variant functions (plus `forward_op_tomo_2d` and `forward_op_tomo_2d_transpose` if unused) into a new `slitless/forward_legacy.py` module. Add deprecation warnings via `warnings.warn()` with `DeprecationWarning` on first import. Update `apj_tomo_plotter.py` to import from the legacy module instead of `forward.py`.

```python
# slitless/forward_legacy.py
"""Deprecated forward model variants. Use slitless.forward instead."""
import warnings
warnings.warn(
    "forward_legacy is deprecated. Use slitless.forward (forward_op_tomo_3d, "
    "forward_op_tomo_3d_transpose) which accept arbitrary spectral orders.",
    DeprecationWarning,
    stacklevel=2
)

# ... variant function bodies unchanged ...
```

`forward.py` shrinks by ~140 lines. The canonical functions remain.

**Note:** Per the developer familiarity rules, the old import paths must still work. But the variants are imported from `slitless.forward`, not from their own module. Since `apj_tomo_plotter.py` is the only consumer and it's not a production script, it's acceptable to update its imports.

**Pros:**
- `forward.py` shrinks from 740 to ~600 lines — cleaner, focused on canonical implementations.
- Deprecated code is quarantined in a dedicated module with clear intent.
- `apj_tomo_plotter.py` still works (just imports from a different module).
- Runtime deprecation warning signals that the variants are not for new use.
- Reversible: if a variant proves unexpectedly necessary, it's one `from slitless.forward_legacy import ...` away.

**Cons:**
- `apj_tomo_plotter.py` needs 1 line changed (import path).
- Running `apj_tomo_plotter.py` will now emit one deprecation warning on stderr. For a validation script, this is acceptable.
- A copy-paste instead of actual code sharing: the variants' logic is mostly superseded by the canonical versions, but they're kept as-is rather than refactored.

### Option C: Delete All Variants Now
**Description:** Remove `_k3`, `_v0`, `_transpose_k3` variants from `forward.py`. Update `apj_tomo_plotter.py` to use only canonical functions (`forward_op_tomo_3d`, `forward_op_tomo_3d_transpose`). Delete `forward_op_tomo_2d` and `forward_op_tomo_2d_transpose` (apparent dead code).

**Pros:**
- Cleanest result — no deprecated code, no legacy module.
- `forward.py` immediately clean.
- Forces the one consumer to modernize.

**Cons:**
- The `_k3` and `_v0` variants may produce subtly different numerical results from the canonical versions (different internal rolling/summing logic). Deleting them before validating equivalence could lose the comparison ability that `apj_tomo_plotter.py` provides.
- If the developer wrote `apj_tomo_plotter.py` to compare variants for a paper figure, deleting them removes the ability to reproduce that figure.
- Higher risk than Option B for minimal additional gain.

### Option D: Keep Everything As-Is
**Description:** Identical to Option A. Listed for completeness.

## Analysis: Variant vs Canonical Differences

To understand the risk of removing variants, note the key differences:

- **`_k3` vs canonical**: `_k3` always uses orders `[0, -1, 1]` and has inline rolling logic. The canonical `forward_op_tomo_3d` accepts arbitrary orders and has the same rolling logic but generalized. For `orders=[0,-1,1]`, they should produce identical results (same algorithm).
- **`_v0` vs canonical**: `_v0` adds ±2nd order support via interpolation (`interp2d`). The canonical version supports ±2nd orders natively. For `orders=[0,-1,1]`, the `_v0` path doesn't activate the ±2nd order code, so it should match `_k3`.
- **`_transpose_k3` vs canonical**: Same pattern — `_k3` variant is hardcoded to 3 orders; canonical is parameterized. Should produce identical results for `orders=[0,-1,1]`.

The variants appear to be evolutionary snapshots: `_k3` → `_v0` → canonical. The canonical versions are strict supersets. `apj_tomo_plotter.py` likely imports both to verify equivalence (it does have "tester" in its lineage).

## Developer Familiarity Impact

| Question | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| `apj_tomo_plotter.py` still runs? | Yes | Yes (1 import changed) | Must update all imports | Yes |
| `forward.py` line count | 740 | ~600 | ~550 | 740 |
| Deprecated code visible when reading forward.py? | Yes | No | No | Yes |
| Can developer still access variants? | Yes (same file) | Yes (different file) | No | Yes |
| Risk of breaking numerical comparison | None | None (variants preserved) | Medium (variants deleted) | None |

## Recommendation

**Option B** (legacy module). The `forward_legacy.py` module requires exactly one change in consumer code (`apj_tomo_plotter.py`'s import line). The variants are preserved for reproducibility, `forward.py` is cleaned, and the deprecation warning is a clear signal to future readers.

**Implementation order:**
1. Create `slitless/forward_legacy.py` containing the 4 variant functions + `forward_op_tomo_2d` and `forward_op_tomo_2d_transpose` (the 2D versions have no known callers and should also be moved).
2. Add `DeprecationWarning` on module import.
3. Update `apj_tomo_plotter.py` to import from `slitless.forward_legacy`.
4. Remove the variant functions from `forward.py`.
5. Verify `apj_tomo_plotter.py` runs and produces identical output (compare figures visually).

**Verification:** Run `apj_tomo_plotter.py` before and after the change. The plots must be identical — the variants' implementations are not being modified, only relocated.
