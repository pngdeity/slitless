"""Smoke tests for slitless package. Run with: pytest tests/test_smoke.py -v"""

import numpy as np
import pytest


def test_forward_import():
    from slitless.forward import forward_op, Source, Imager
    assert True


def test_config_import():
    from slitless.config import SlitlessConfig, config
    assert True


def test_measure_import():
    from slitless.measure import compare_ssim, nrmse, compare_psnr
    assert True


def test_data_loader_import():
    from slitless.data_loader import BasicDataset
    assert True


def test_common_import():
    from slitless.common import outch_adjuster
    assert True


def test_plotting_import():
    from slitless.plotting import uiuc_im
    assert True


def test_unet_import():
    from slitless.networks.unet import UNet
    assert True


def test_solvers_import():
    from slitless.solvers._tomographic import smart, tomoinv
    from slitless.solvers._optimization import scipy_solver
    from slitless.solvers._prior import prior_solver
    assert True


def test_config_fields():
    from slitless.config import SlitlessConfig
    c = SlitlessConfig()
    assert c.intensity_scale == 6000.0
    assert c.lamdim == 64
    assert c.patch_size == 64
    assert c.wavelength == 195.117937907451


def test_ssim_batched():
    from slitless.measure import compare_ssim
    a = np.random.rand(4, 3, 64, 64)
    b = np.random.rand(4, 3, 64, 64)
    result = compare_ssim(truth=a, estimate=b)
    assert result.shape == (4, 3)


def test_nrmse_batched():
    from slitless.measure import nrmse
    a = np.random.rand(4, 3, 64, 64)
    b = np.random.rand(4, 3, 64, 64)
    result = nrmse(truth=a, estimate=b)
    assert result.shape == (4, 3)


def test_recon_facade_imports():
    from slitless.recon import (
        Reconstructor, Reconstructor_Multi, Recon,
        scipy_solver, smart, nn_solver, grad_descent_solver,
        tomoinv, prior_solver,
        smart2, scipy_solver_parallel, scipy_solver_parallel2,
        gauss_pmf_fitter, gauss_pmf_fitter2,
        comparison_test_multi, SOLVERS,
    )
    assert 'scipy' in SOLVERS
    assert 'smart' in SOLVERS
    assert 'nn' in SOLVERS


def test_forward_legacy():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        from slitless.forward_legacy import forward_op_tomo_3d_k3
    assert True


def test_forward_op_golden():
    import os
    from slitless.forward import Source, Imager
    HERE = os.path.dirname(__file__)
    golden_dir = os.path.join(HERE, 'golden')
    if not os.path.exists(os.path.join(golden_dir, 'intensity.npy')):
        pytest.skip("Golden data not available")
    intensity = np.load(os.path.join(golden_dir, 'intensity.npy'))
    velocity = np.load(os.path.join(golden_dir, 'velocity.npy'))
    linewidth = np.load(os.path.join(golden_dir, 'linewidth.npy'))
    expected = np.load(os.path.join(golden_dir, 'meas.npy'))
    source = Source(inten=intensity, vel=velocity, width=linewidth)
    imager = Imager(
        pixel_size=1,
        dispersion=0.022275,
        spectral_orders=[0, -1, 1],
        pixelated=True,
        dbsnr=0,
        noise_model='poisson',
    )
    result = imager.forward_op(intensity, velocity, linewidth)
    assert np.allclose(result, expected, rtol=1e-10), \
        f"Max diff: {np.max(np.abs(result - expected))}"
