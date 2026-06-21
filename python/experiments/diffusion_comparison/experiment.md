# Diffusion Comparison Experiment

## Goal

Compare three reconstruction approaches on the slitless spectroscopy inverse
problem: a supervised U-Net baseline, diffusion posterior sampling via DPS
(unconditional diffusion model + measurement guidance), and a conditional
diffusion model (CondDiff). All methods are evaluated at K=3 detectors under
three noise conditions: noiseless (γ=∞), 30 dB Gaussian SNR, and 20 dB
Gaussian SNR.

## Setup

- **Dataset**: `eis_test_50_dsetv6.npy` — 20 independent 64×64 patches from
  the EIS test split (dset_v6, leakage-free partition). Parameters are in
  physical units (intensity in erg/cm²/s/sr, velocity in km/s, width in Å).
- **Detectors**: K=3, orders {0, −1, +1}.
- **Noise model**: Gaussian, added on top of the noiseless pre-computed
  measurements stored in the dataset.

### Models

| Method   | γ=∞ model                                               | γ=30 dB model                                   | γ=20 dB model                                   |
|----------|---------------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| U-Net    | `2026_05_28__00_09_08_..._None_K_3_dset_v6_logzscale`  | `2026_05_31__12_00_32_..._30_gaussian_K_3_...`  | `2026_05_31__13_07_15_..._20_gaussian_K_3_...`  |
| DPS      | `run_all_lr_1e-4_cosine_b32_logz/model-10.pt` (shared) | same                                            | same                                            |
| CondDiff | `run_all_lr1e-4_cosine_b32_conditional_logz/model-10.pt` | `2026_06_02__21_03_55_..._Gaussian_30/model-10.pt` | `2026_05_31__04_12_50_..._Gaussian_20/model-10.pt` |

## Results

### RMSE / Bias Table

*(Fill in from `output/metrics_summary.txt` after running.)*

| γ (dB) | Method   | Int RMSE | Int Bias | Vel RMSE | Vel Bias | Wid RMSE | Wid Bias |
|--------|----------|----------|----------|----------|----------|----------|----------|
| ∞      | U-Net    |          |          |          |          |          |          |
| ∞      | DPS      |          |          |          |          |          |          |
| ∞      | CondDiff |          |          |          |          |          |          |
| 30     | U-Net    |          |          |          |          |          |          |
| 30     | DPS      |          |          |          |          |          |          |
| 30     | CondDiff |          |          |          |          |          |          |
| 20     | U-Net    |          |          |          |          |          |          |
| 20     | DPS      |          |          |          |          |          |          |
| 20     | CondDiff |          |          |          |          |          |          |

### Spatial Reconstructions

*(See `output/recon_comp_dbsnr_*.png`)*

Key things to inspect:
- Does CondDiff preserve spatial morphology better than DPS?
- Do diffusion methods introduce hallucinated structure in velocity / line width?
- How does DPS trade off between measurement consistency and prior-driven smoothness?

### Scatter Plots (True vs. Estimated)

*(See `output/scatter_dbsnr_*.png`)*

Pearson r across all test patches:

| γ (dB) | Method   | r (Int) | r (Vel) | r (Wid) |
|--------|----------|---------|---------|---------|
| ∞      | U-Net    |         |         |         |
| ∞      | DPS      |         |         |         |
| ∞      | CondDiff |         |         |         |
| 30     | U-Net    |         |         |         |
| 30     | DPS      |         |         |         |
| 30     | CondDiff |         |         |         |
| 20     | U-Net    |         |         |         |
| 20     | DPS      |         |         |         |
| 20     | CondDiff |         |         |         |

## Discussion

*(Fill in after examining the output.)*

**Velocity recovery**: The supervised U-Net was shown in the APJ study to
reduce velocity RMSE by roughly a factor of 2–3 over MART and 1D MAP.
The question here is whether the diffusion prior — especially when applied
conditionally — can match or exceed the U-Net's velocity accuracy, since
diffusion models learn a richer prior over the joint (intensity, velocity,
line-width) distribution.

**Line-width recovery**: This is the most challenging parameter. Classical
methods (MART, 1D MAP) fail to track spatial variation; the U-Net achieves
r≈0.72. Diffusion models trained directly on the parameter space may learn
the line-width distribution more explicitly, but DPS relies on measurement
guidance gradients that could be weak for line width if the forward operator
is insensitive to small width changes.

**DPS vs. CondDiff**: DPS couples the unconditional diffusion prior to the
measurements through gradient guidance at each sampling step. CondDiff
injects the measurement as a conditioning input, enabling the network to
learn a direct posterior approximation. CondDiff is expected to be faster
(DDIM sampling with 250 steps vs. DPS's 1000 full steps) and may achieve
better calibration at the target SNR, since each conditional model was
trained specifically for that noise level. However, DPS gains flexibility
by not requiring noise-level-specific training.

**Noise robustness**: U-Net models trained with matched Gaussian noise are
expected to be strong baselines at the trained SNR. Whether CondDiff at
γ=20 dB surpasses the matched U-Net — or merely approaches it — will
indicate whether the richer diffusion prior justifies the additional
complexity and computation.

## Conclusions

*(Fill in after examining the output.)*

## Outputs

```
output/
  Rec_unet_dbsnr_None_None.pickle
  Rec_dps_dbsnr_None_None.pickle
  Rec_cond_dbsnr_None_None.pickle
  Rec_unet_dbsnr_20_gaussian.pickle
  Rec_dps_dbsnr_20_gaussian.pickle
  Rec_cond_dbsnr_20_gaussian.pickle
  Rec_unet_dbsnr_30_gaussian.pickle
  Rec_dps_dbsnr_30_gaussian.pickle
  Rec_cond_dbsnr_30_gaussian.pickle
  results_summary.pickle          ← from runner.py
  recon_comp_dbsnr_None_None.png
  recon_comp_dbsnr_20_gaussian.png
  recon_comp_dbsnr_30_gaussian.png
  scatter_dbsnr_None_None.png
  scatter_dbsnr_20_gaussian.png
  scatter_dbsnr_30_gaussian.png
  bar_charts.png
  table.tex
  metrics_summary.txt
```
