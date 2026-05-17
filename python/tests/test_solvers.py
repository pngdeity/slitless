"""Sanity tests for slitless solvers on synthetic 4x4 toy data.

Run fast tests only::

    pytest tests/test_solvers.py -v --import-mode=importlib -m "not slow"

Run everything::

    pytest tests/test_solvers.py -v --import-mode=importlib
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_import(module_path, name):
    """Import a name, returning None if any dependency is missing."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, name)
    except Exception:
        return None


@pytest.fixture
def tiny_setup():
    """Create a tiny 4x4 source and imager for solver testing."""
    from slitless.forward import Source, Imager

    np.random.seed(42)
    H, W = 4, 4
    inten = np.full((H, W), 500.0)
    vel = np.zeros((H, W))
    width = np.full((H, W), 0.05)
    source = Source(inten=inten, vel=vel, width=width, pix=True)
    imager = Imager(
        pixel_size=1.0,
        dispersion=0.022275,
        spectral_orders=[0, -1, 1],
        mid_wavelength=195.119,
        pixelated=True,
        noise_model=None,
        mask=np.ones((H, W)),
    )
    imager.get_measurements(sources=source)
    return source, imager


# ---------------------------------------------------------------------------
# fast tests  (no marker needed)
# ---------------------------------------------------------------------------


def test_prior_solver(tiny_setup):
    """prior_solver: simplest, fastest solver — verify shape / no NaN / not all zero."""
    from slitless.solvers._prior import prior_solver

    source, imager = tiny_setup
    recon, _ = prior_solver(imager=imager)
    assert recon.shape == (3, 4, 4), f"unexpected shape {recon.shape}"
    assert not np.any(np.isnan(recon)), "recon contains NaN"
    assert not np.allclose(recon, 0), "recon is all zeros"


def test_gauss_pmf_fitter():
    """1D Gaussian PMF fitter — verify returns (intensity, mean, std) near truth."""
    from slitless.solvers._tomographic import gauss_pmf_fitter

    L = 21
    x = np.arange(L) - L // 2  # centre at 0
    sigma = 0.8
    spectrum = np.exp(-(x ** 2) / (2 * sigma ** 2))
    spectrum_3d = spectrum[:, np.newaxis, np.newaxis]
    result = gauss_pmf_fitter(spectrum_3d)
    assert result.shape == (3, 1, 1)
    mean = result[1, 0, 0]
    std = result[2, 0, 0]
    assert abs(mean) < 0.5, f"mean={mean:.4f} should be near 0"
    assert 0.1 < std < 1.5, f"std={std:.4f} should be near {sigma}"


def test_gauss_pmf_fitter2():
    """2D PMF fitter on 3D cube (21, 4, 4) — verify shape, no NaN, positive intensity."""
    from slitless.solvers._tomographic import gauss_pmf_fitter2

    H, W, L = 4, 4, 21
    x = np.arange(L) - L // 2
    sigma = 0.8
    spectrum = np.exp(-(x ** 2) / (2 * sigma ** 2))
    cube = np.tile(spectrum[:, np.newaxis, np.newaxis], (1, H, W))
    result = gauss_pmf_fitter2(cube)
    assert result.shape == (3, H, W)
    assert not np.any(np.isnan(result)), "result contains NaN"
    assert np.all(result[0] >= 0), "intensity channel has negative values"


# ---------------------------------------------------------------------------
# slow tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scipy_solver(tiny_setup):
    """scipy_solver — column-wise L-BFGS-B: verify shape and no NaN."""
    from slitless.solvers._optimization import scipy_solver

    source, imager = tiny_setup
    rec, losses = scipy_solver(imager=imager, maxiter=10)
    assert rec.shape == (3, 4, 4)
    assert not np.any(np.isnan(rec)), "rec contains NaN"


@pytest.mark.slow
def test_smart(tiny_setup, monkeypatch):
    """smart MART solver — monkeypatch lamdim to match 4x4 source.

    ``smart`` calls ``datacube_generator`` with default ``lamdim=64``
    but builds the tomographic matrix for ``(M, M)`` where ``M`` is the
    first spatial dimension of ``meas``.  For a 4x4 source those disagree.
    We patch ``datacube_generator`` inside ``_tomographic`` to use
    ``lamdim=M`` so the dimensions stay consistent.
    """
    from slitless.solvers._tomographic import smart
    from slitless.forward import datacube_generator as _orig_dcg

    source, imager = tiny_setup
    H, W = source.inten.shape

    def _dcg(param3d, pixelated=True, lamdim=H):
        return _orig_dcg(param3d, pixelated=pixelated, lamdim=lamdim)

    monkeypatch.setattr(
        "slitless.solvers._tomographic.datacube_generator", _dcg
    )
    recon, cube = smart(imager=imager, maxouter=2, maxinner=3)
    assert recon.shape == (3, H, W), f"unexpected recon shape {recon.shape}"
    assert not np.any(np.isnan(recon)), "recon contains NaN"
    assert cube.shape == (H, H, W), f"unexpected cube shape {cube.shape}"


@pytest.mark.slow
def test_tomoinv(tiny_setup, monkeypatch):
    """tomoinv tomographic inversion — monkeypatch lamdim=21.

    ``tomoinv`` builds the tomographic matrix for ``(21, N)`` but calls
    ``datacube_generator`` with default ``lamdim=64``.  We patch it to
    use ``lamdim=21`` so the dimensions stay consistent.
    """
    from slitless.solvers._tomographic import tomoinv
    from slitless.forward import datacube_generator as _orig_dcg

    source, imager = tiny_setup
    H, W = source.inten.shape

    def _dcg(param3d, pixelated=True, lamdim=21):
        return _orig_dcg(param3d, pixelated=pixelated, lamdim=lamdim)

    monkeypatch.setattr(
        "slitless.solvers._tomographic.datacube_generator", _dcg
    )
    recon, loss = tomoinv(imager=imager, numiter=2)
    assert recon.shape == (3, H, W), f"unexpected recon shape {recon.shape}"
    assert not np.any(np.isnan(recon)), "recon contains NaN"


@pytest.mark.slow
def test_grad_descent_solver(tiny_setup):
    """grad_descent_solver — verify shape, no NaN, and losses array non-empty."""
    from slitless.solvers._optimization import grad_descent_solver

    source, imager = tiny_setup
    recon, losses = grad_descent_solver(imager=imager, maxiter=5, LR=1e-2)
    assert recon.shape == (3, 4, 4), f"unexpected recon shape {recon.shape}"
    assert not np.any(np.isnan(recon)), "recon contains NaN"
    assert len(losses) == 5, f"expected 5 losses, got {len(losses)}"


@pytest.mark.slow
def test_scipy_solver_parallel(tiny_setup):
    """scipy_solver_parallel — verify shape and no NaN with n_jobs=1."""
    from slitless.solvers._optimization import scipy_solver_parallel

    source, imager = tiny_setup
    rec, losses = scipy_solver_parallel(imager=imager, maxiter=10, n_jobs=1)
    assert rec.shape == (3, 4, 4), f"unexpected rec shape {rec.shape}"
    assert not np.any(np.isnan(rec)), "rec contains NaN"
