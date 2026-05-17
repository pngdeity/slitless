"""Integration tests for slitless end-to-end pipeline and consistency checks."""

import os
import numpy as np


def test_full_pipeline_scipy():
    """Forward model -> noise -> scipy_solver -> verify RMSE below threshold."""
    from slitless.forward import Source, Imager
    from slitless.recon import scipy_solver

    np.random.seed(42)
    H, W = 8, 8
    true_inten = np.random.uniform(200, 800, (H, W))
    true_vel = np.random.uniform(-0.5, 0.5, (H, W))
    true_width = np.random.uniform(0.5, 1.5, (H, W))
    true_param = np.stack([true_inten, true_vel, true_width], axis=0)

    source = Source(inten=true_inten, vel=true_vel, width=true_width, pix=True)

    imager = Imager(
        pixel_size=1.0,
        dispersion=0.022275,
        spectral_orders=[0, -1, 1],
        mid_wavelength=195.119,
        pixelated=True,
        mask=np.ones((H, W)),
        noise_model=None,
    )
    imager.get_measurements(sources=source, noise_model=None)

    recon, losses = scipy_solver(
        imager=imager, maxiter=500, lam_i=0, lam_v=0, lam_w=0
    )

    assert recon.shape == (3, H, W), f"Expected (3,{H},{W}), got {recon.shape}"
    assert not np.any(np.isnan(recon)), "NaN in reconstruction"

    rmse = np.sqrt(np.mean((recon - true_param) ** 2, axis=(1, 2)))
    assert rmse[0] < 50, f"Intensity RMSE {rmse[0]:.1f} too high"
    assert rmse[1] < 0.1, f"Velocity RMSE {rmse[1]:.4f} too high"


def test_config_values_consistent():
    """Verify config defaults produce same behavior as hardcoded values."""
    from slitless.config import SlitlessConfig

    c = SlitlessConfig()

    assert c.intensity_scale == 6000.0
    assert c.wavelength == 195.117937907451
    assert c.speed_of_light == 299792.458
    assert c.mid_wavelength == 195.119
    assert c.dispersion_scale == 0.022275
    assert c.numdetectors == 3


def test_metrics_identical_input():
    """SSIM should be 1.0 for identical images, NRMSE should be 0.0."""
    from slitless.measure import compare_ssim, nrmse

    a = np.random.rand(64, 64).astype(np.float64)

    ssim_val = compare_ssim(truth=a, estimate=a)
    assert np.allclose(ssim_val, 1.0, atol=1e-6), f"SSIM={ssim_val}"

    nrmse_val = nrmse(truth=a, estimate=a)
    assert np.allclose(nrmse_val, 0.0, atol=1e-6), f"NRMSE={nrmse_val}"


def test_solvers_registry():
    """Verify SOLVERS dict maps all registered solvers to callable functions."""
    from slitless.recon import SOLVERS

    assert len(SOLVERS) >= 10, f"Expected >=10 solvers, got {len(SOLVERS)}"

    for name, func in SOLVERS.items():
        assert callable(func), f"SOLVERS['{name}'] is not callable: {type(func)}"


def test_reconstructor_dispatch():
    """Verify Reconstructor can dispatch a solver and produce a Recon object."""
    from slitless.forward import Source, Imager
    from slitless.recon import Reconstructor, scipy_solver, Recon

    H, W = 8, 8
    inten = np.full((H, W), 500.0)
    vel = np.zeros((H, W))
    width = np.full((H, W), 1.0)
    param = np.stack([inten, vel, width], axis=0)
    source = Source(inten=inten, vel=vel, width=width, pix=True)

    imager = Imager(
        pixel_size=1.0,
        dispersion=0.022275,
        spectral_orders=[0, -1, 1],
        mid_wavelength=195.119,
        pixelated=True,
        mask=np.ones((H, W)),
        noise_model=None,
    )
    imager.get_measurements(sources=source, noise_model=None)

    rec = Reconstructor(
        imager=imager,
        solver=scipy_solver,
        maxiter=20,
        lam_i=1e-2,
        lam_v=1e-2,
        lam_w=1e-2,
        simulate_meas=False,
    )
    recon = rec.solve()
    assert isinstance(recon, Recon), f"Expected Recon, got {type(recon)}"


def test_production_scripts_exist():
    """Verify the developer's production scripts still exist at expected paths."""
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    required = [
        "eis_reader_v3.py",
        "generate_dset_v5.py",
        "final_result_runner.py",
        "check_v5_stats.py",
    ]

    for script in required:
        path = os.path.join(scripts_dir, script)
        assert os.path.exists(path), f"Missing production script: {script}"
