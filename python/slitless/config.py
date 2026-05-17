"""Central configuration for slitless spectral imaging."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


def _default_data_root() -> Path:
    env = os.environ.get('SLITLESS_DATA', '')
    if env:
        return Path(env)
    return Path.home() / 'resources' / 'slitless' / 'data'


def _default_results_root() -> Path:
    env = os.environ.get('SLITLESS_RESULTS', '')
    if env:
        return Path(env)
    return Path.home() / 'resources' / 'slitless' / 'python' / 'results'


@dataclass
class SlitlessConfig:
    """Master configuration for slitless spectral imaging pipeline."""

    # --- Paths ---
    data_root: Path = field(default_factory=_default_data_root)
    results_root: Path = field(default_factory=_default_results_root)

    @property
    def norm_stats_path(self) -> Path:
        return self.data_root / 'eis_data' / 'datasets' / 'dset_v5' / 'norm_stats.npy'

    @property
    def template_path(self) -> Path:
        return self.data_root / 'eis_data' / 'templates' / 'fe_12_195_119.2c.template.h5'

    @property
    def train_data_dir(self) -> Path:
        return self.data_root / 'eis_data' / 'datasets' / 'dset_v5' / 'data'

    @property
    def model_dir(self) -> Path:
        return self.results_root / 'saved'

    @property
    def diffusion_model_dir(self) -> Path:
        return Path.home() / 'resources' / 'denoising-diffusion-pytorch' / 'results'

    # --- Physics Constants ---
    wavelength: float = 195.117937907451        # Fe XII 195.12 Angstroms
    speed_of_light: float = 299792.458          # km/s
    mid_wavelength: float = 195.119             # Angstroms
    dispersion_scale: float = 0.022275          # Angstroms per pixel

    # --- Instrument ---
    pixel_size: float = 1.0                     # arcsec
    spectral_orders: List[int] = field(default_factory=lambda: [0, -1, 1])

    # --- Normalization ---
    intensity_scale: float = 6000.0             # 99.9th percentile
    # NOTE: velocity/linewidth normalization stats (mean, std) are loaded at
    # runtime from norm_stats.npy via data_loader._get_stats(). Typical values:
    #   velocity_mean ≈ 0.0, velocity_std ≈ 2.0 (pixels)
    #   linewidth_mean ≈ 0.0, linewidth_std ≈ 2.0 (pixels)

    # --- Dataset ---
    lamdim: int = 64          # Wavelength dimension (spectral pixels)
    patch_size: int = 64      # Spatial crop size for dataset generation

    # --- Training ---
    numdetectors: int = 3
    learning_rate: float = 2e-4
    epochs: int = 200
    batch_size: int = 4
    bilinear: bool = True
    num_filters: int = 16
    outch_type: str = 'all'                     # 'int', 'vel', 'width', or 'all'
    optimizer_type: str = 'adam'
    loss_type: str = 'mse'

    # --- Noise ---
    noise_model: str = 'poisson'
    dbsnr: float = 0.0                          # 0 = no noise; positive = SNR in dB

    # --- Device ---
    device: str = 'cuda'                        # 'cuda' or 'cpu'


# Singleton instance for backward compatibility
config = SlitlessConfig()
