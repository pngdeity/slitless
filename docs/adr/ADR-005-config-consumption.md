# ADR-005: Config Consumption Gaps

**Status:** Proposed
**Date:** 2026-05-17

## Context

`config.py` defines a `SlitlessConfig` dataclass with 23 fields across 7 categories (paths, physics, instrument, normalization, training, noise, device). A singleton `config` instance is exported for backward compatibility.

However, 6 config fields are defined but **never consumed** by library code:

| Field | Value | Defined? | Consumed by library? | Where it's actually needed |
|---|---|---|---|---|
| `intensity_scale` | `6000.0` | config.py:61 | **No** — `data_loader.meas_transform` hardcodes `6000` on lines 18, 21 | `data_loader.py` |
| `velocity_mean` | `0.0` | config.py:62 | **No** — values come from `norm_stats.npy` at runtime | `data_loader.py` (param_transform) |
| `velocity_std` | `1.0` | config.py:63 | **No** | `data_loader.py` (param_transform) |
| `linewidth_mean` | `0.0` | config.py:64 | **No** | `data_loader.py` (param_transform) |
| `linewidth_std` | `1.0` | config.py:65 | **No** | `data_loader.py` (param_transform) |
| `numdetectors` | `3` | config.py:68 | **Yes** — `train.py:131` reads it | N/A (already consumed) |

Fields that are consumed (for reference): `speed_of_light`, `mid_wavelength`, `dispersion_scale`, `wavelength` (all used in `recon.py` for unit conversion); `data_root`, `results_root`, `norm_stats_path`, `template_path`, `train_data_dir`, `model_dir`, `diffusion_model_dir` (used for path construction); `learning_rate`, `epochs`, `batch_size`, `bilinear`, `num_filters`, `outch_type`, `optimizer_type`, `loss_type` (used in `train.py`); `noise_model`, `dbsnr` (used in `forward.py`); `device` (used in `train.py` and others).

Additionally, values **missing** from config that are hardcoded in library code or scripts:

| Value | Hardcoded location | Current value | Notes |
|---|---|---|---|
| `lamdim` (wavelength dimension) | `forward.py:79`, `eistools.py:233`, `generate_dset_v5.py:14`, `recon.py` (multiple) | `64` (library default), `21` (v5 dataset) | Should be configurable per-Imager |
| `patch_size` | `eistools.py` (random cropper), `generate_dset_v5.py` | `64` | Dataset generation parameter |
| `dataset_version` | `generate_dset_v5.py:9` (path string) | `v5` | Path construction in multiple files |

## Decision Options

### Option A: Status Quo
**Description:** Keep 5 normalization fields in config as documentation-only defaults. Keep hardcoded `6000` in `data_loader.py`. Keep `lamdim`, `patch_size`, `dataset_version` as hardcoded constants.

**Pros:**
- Zero migration cost.
- The config values serve as documentation of what the "correct" defaults are, even if not wired to consumers.

**Cons:**
- **Dual source of truth**: changing normalization from `6000` to `7000` requires editing both `config.py` and `data_loader.py`. If only one is changed, behavior diverges silently.
- Config fields that claim to control normalization don't actually control normalization — misleading.
- Script-level hardcoded values (`lamdim=21` in `generate_dset_v5.py` vs `lamdim=64` default in `forward.py`) create confusing inconsistencies.

### Option B: Wire All Fields + Add Missing Ones (Recommended)
**Description:**
1. Wire `intensity_scale` to `data_loader.meas_transform` and `meas_inv_transform` (replace hardcoded `6000`). Accomplished automatically by ADR-003's lazy loading change.
2. Remove `velocity_mean`, `velocity_std`, `linewidth_mean`, `linewidth_std` from config — they are runtime data loaded from `norm_stats.npy`, not configuration. Keep them as documentation comments only.
3. Add `lamdim: int = 64` and `patch_size: int = 64` to `SlitlessConfig`.
4. Add `dataset_version: str = "v5"` to config so path properties can reference it instead of hardcoding version strings.

```python
# config.py changes
@dataclass
class SlitlessConfig:
    # --- Normalization ---
    intensity_scale: float = 6000.0             # Single source of truth for scaling
    # velocity/linewidth mean/std are runtime-loaded from norm_stats.npy, not config

    # --- Dataset ---
    lamdim: int = 64                            # Wavelength dimension (spectral pixels)
    patch_size: int = 64                        # Spatial crop size for dataset generation
    dataset_version: str = "v5"                 # Active dataset version for path construction
```

**Consumed in:**
- `intensity_scale` → `data_loader.meas_transform`, `param_transform`, `param_inv_transform`
- `lamdim` → `forward.datacube_generator` (default parameter), `eistools.eis_to_ssi_interpolator` (default parameter)
- `patch_size` → `eistools.eis_random_cropper2` (default parameter)
- `dataset_version` → `config.norm_stats_path`, `config.train_data_dir` (replace hardcoded `dset_v5` with `f'dset_{self.dataset_version}'`)

**Pros:**
- Single source of truth for `intensity_scale` — no more silent divergence between config and data_loader.
- `lamdim` and `patch_size` can be changed in one place (`config.py`) instead of hunting through 5+ files.
- Config path properties become version-aware automatically.
- `velocity_mean`/`velocity_std`/`linewidth_mean`/`linewidth_std` removed from config eliminates confusion about their role (they are not config, they are data).

**Cons:**
- Changes `data_loader.meas_transform` behavior — if `config.intensity_scale` is ever changed from `6000`, the transform changes. This is intentional (config should be the source of truth).
- Default `lamdim=64` in config differs from `lamdim=21` currently used by `generate_dset_v5.py`. Scripts that import config's default will get 64 unless they pass `lamdim` explicitly. This is correct behavior — the script should override the default.
- Path properties that embed `dataset_version` change dynamically if `version` changes — could be surprising if someone changes `version` without moving data files.

### Option C: Remove Unused Config Fields
**Description:** Delete `intensity_scale`, `velocity_mean`, `velocity_std`, `linewidth_mean`, `linewidth_std` from config. Keep hardcoded values where they are. No new fields added.

**Pros:**
- Config is truthful — every field is consumed.
- No behavioral changes.

**Cons:**
- Does not solve the dual-source-of-truth problem.
- Adds friction: if hardcoded values need changing, developer must edit multiple files.
- Misses the opportunity to add `lamdim`/`patch_size` as configurable parameters.

## Developer Familiarity Impact

| Question | Option A | Option B | Option C |
|---|---|---|---|
| `config.intensity_scale` changes → `meas_transform` changes? | No (disconnected) | Yes (wired) | N/A (field removed) |
| Changing normalization requires editing... | 2 files | 1 file (`config.py`) | 1 file (`data_loader.py`) |
| `lamdim` change requires editing... | 4+ files | 1 file (or pass override) | 4+ files |
| `velocity_mean` etc. removed from config? | No | Yes | Yes |
| Existing behavior changes? | No | Only if `intensity_scale` is changed from 6000 | No |

## Recommendation

**Option B** with these specific deltas:

1. **Wire `intensity_scale`**: The `data_loader` transforms should read `config.intensity_scale` instead of hardcoding `6000`. Since the default is also `6000`, behavior is unchanged unless someone intentionally changes `intensity_scale` in config.

2. **Remove normalization stats from config**: `velocity_mean`, `velocity_std`, `linewidth_mean`, `linewidth_std` are not configuration — they are summary statistics of a specific dataset, loaded at runtime from `norm_stats.npy`. Config should not pretend to control them. Replace with a comment block documenting their typical values.

3. **Add `lamdim`**: Configurable at 64 (matching the Imager default), overridable per-call. This doesn't change behavior but provides a central place to set it.

4. **Add `patch_size`**: Same pattern — default matches current behavior, overridable.

5. **Defer `dataset_version`**: The version is embedded in directory paths (`dset_v5`). Making it config-controlled risks path breakage if changed without moving data. Defer this until path management is more mature (e.g., a path registry).

**Critical invariant**: After wiring, `config.intensity_scale = 6000` must produce identical behavior to today's hardcoded `6000` in `meas_transform`. This can be verified by running a single forward-measurement round-trip before and after the change and asserting the output is bitwise identical.
