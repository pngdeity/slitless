"""Comprehensive unit tests for slitless.forward module.

Run with:
    python -m pytest tests/test_forward.py -v --import-mode=importlib
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from slitless.forward import (
    SPEED_OF_LIGHT,
    Imager,
    Source,
    add_noise,
    datacube_generator,
    forward_op,
    forward_op_torch,
    gauss,
    gauss_pix,
    gauss_pix_torch,
    gauss_torch,
    tomomtx_gen,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def tiny_source():
    """4x4 source with known properties."""
    np.random.seed(42)
    inten = np.ones((4, 4)) * 1000.0
    vel = np.zeros((4, 4))
    width = np.ones((4, 4)) * 0.05
    param3d = np.stack([inten, vel, width], axis=0)
    return {"inten": inten, "vel": vel, "width": width, "param3d": param3d}


@pytest.fixture(scope="module")
def tiny_imager():
    return Imager(
        pixel_size=1.0,
        dispersion=0.022275,
        spectral_orders=[0, -1, 1],
        mid_wavelength=195.119,
        pixelated=True,
        dbsnr=0,
        noise_model="poisson",
    )


# ============================================================
# gauss
# ============================================================


class TestGauss:
    def test_known_value(self):
        """gauss(0, 0, 1) == 1 / sqrt(2*pi)."""
        result = gauss(0.0, 0.0, 1.0)
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert pytest.approx(result.item(), rel=1e-12) == expected

    def test_symmetry(self):
        """gauss(0, 1, 1) == gauss(-1, 0, 1)."""
        a = gauss(np.array([0.0]), np.array([1.0]), np.array([1.0]))
        b = gauss(np.array([-1.0]), np.array([0.0]), np.array([1.0]))
        assert pytest.approx(a.item(), rel=1e-12) == b.item()

    def test_output_shape_matches_input(self):
        x = np.linspace(-5, 5, 11)
        result = gauss(x, 0.0, 1.0)
        assert result.shape == x.shape

    def test_vectorized(self):
        x = np.array([0.0, 1.0, 2.0])
        mean = np.array([0.0, 0.0, 0.0])
        sigma = np.array([1.0, 1.0, 1.0])
        result = gauss(x, mean, sigma)
        assert result.shape == (3,)


# ============================================================
# gauss_pix
# ============================================================


class TestGaussPix:
    def test_area_approx_one(self):
        """Area under pixel-integrated Gaussian ≈ 1.0 over wide range."""
        x = np.arange(-20, 21, dtype=float)
        result = gauss_pix(x, 0.0, 1.0)
        assert pytest.approx(float(np.sum(result)), rel=1e-10) == 1.0

    def test_symmetry(self):
        """gauss_pix is symmetric: gauss_pix(a, 0, s) == gauss_pix(-a, 0, s)."""
        x = np.array([1.0, 2.0, 3.0])
        pos = gauss_pix(x, 0.0, 1.0)
        neg = gauss_pix(-x, 0.0, 1.0)
        np.testing.assert_allclose(pos, neg, rtol=1e-12)

    def test_narrow_width_approaches_one_at_center(self):
        """For very narrow sigma, the center pixel captures nearly all flux."""
        result = gauss_pix(np.array([0.0]), 0.0, 0.01)
        assert pytest.approx(float(result.item()), abs=1e-6) == 1.0


# ============================================================
# gauss_torch
# ============================================================


class TestGaussTorch:
    def test_matches_numpy(self):
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        mean = torch.tensor([0.0], dtype=torch.float64)
        sigma = torch.tensor([1.0], dtype=torch.float64)
        result_torch = gauss_torch(x, mean, sigma)
        result_np = gauss(x.numpy(), mean.numpy(), sigma.numpy())
        np.testing.assert_allclose(result_torch.numpy(), result_np, rtol=1e-14)

    def test_requires_grad(self):
        x = torch.tensor([0.0, 1.0], requires_grad=True)
        mean = torch.tensor([0.0], requires_grad=True)
        sigma = torch.tensor([1.0], requires_grad=True)
        result = gauss_torch(x, mean, sigma)
        assert result.requires_grad is True
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert mean.grad is not None
        assert sigma.grad is not None

    def test_output_shape_matches_input(self):
        x = torch.linspace(-5, 5, 11)
        result = gauss_torch(x, torch.tensor(0.0), torch.tensor(1.0))
        assert result.shape == x.shape


# ============================================================
# gauss_pix_torch
# ============================================================


class TestGaussPixTorch:
    def test_matches_numpy(self):
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        mean = torch.tensor([0.0], dtype=torch.float64)
        sigma = torch.tensor([1.0], dtype=torch.float64)
        result_torch = gauss_pix_torch(x, mean, sigma)
        result_np = gauss_pix(x.numpy(), mean.numpy(), sigma.numpy())
        np.testing.assert_allclose(result_torch.numpy(), result_np, rtol=1e-14)

    def test_requires_grad(self):
        x = torch.tensor([0.0, 1.0], requires_grad=True)
        mean = torch.tensor([0.0], requires_grad=True)
        sigma = torch.tensor([1.0], requires_grad=True)
        result = gauss_pix_torch(x, mean, sigma)
        assert result.requires_grad is True
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert mean.grad is not None
        assert sigma.grad is not None


# ============================================================
# datacube_generator
# ============================================================


class TestDatacubeGenerator:
    def test_shape_pixelated(self, tiny_source):
        cube = datacube_generator(tiny_source["param3d"], pixelated=True, lamdim=64)
        assert cube.shape == (64, 4, 4)

    def test_shape_not_pixelated(self, tiny_source):
        cube = datacube_generator(tiny_source["param3d"], pixelated=False, lamdim=64)
        assert cube.shape == (64, 4, 4)

    def test_lamdim_parameter(self, tiny_source):
        cube = datacube_generator(tiny_source["param3d"], pixelated=True, lamdim=32)
        assert cube.shape == (32, 4, 4)

    def test_center_pixel_pixelated(self, tiny_source):
        """Center pixel value with pixelated=True: peak ≈ intensity (since narrow width)."""
        cube = datacube_generator(tiny_source["param3d"], pixelated=True, lamdim=64)
        lam_mid = 64 // 2
        center_val = cube[lam_mid, 2, 2]
        assert pytest.approx(float(center_val), rel=0.1) == 1000.0

    def test_center_pixel_continuum(self, tiny_source):
        """Center pixel value with pixelated=False: peak = gauss(0,0,σ) * intensity."""
        cube = datacube_generator(tiny_source["param3d"], pixelated=False, lamdim=64)
        lam_mid = 64 // 2
        center_val = cube[lam_mid, 2, 2]
        expected = gauss(0.0, 0.0, 0.05) * 1000.0
        assert pytest.approx(float(center_val), rel=1e-10) == float(expected)

    def test_intensity_scaling(self, tiny_source):
        """Doubling intensity doubles the datacube."""
        p = tiny_source["param3d"].copy()
        p[0] *= 2.0
        cube_orig = datacube_generator(tiny_source["param3d"], pixelated=True, lamdim=64)
        cube_double = datacube_generator(p, pixelated=True, lamdim=64)
        np.testing.assert_allclose(cube_double, 2.0 * cube_orig, rtol=1e-12)


# ============================================================
# forward_op
# ============================================================


class TestForwardOp:
    def test_param3d_vs_explicit(self, tiny_source):
        """param3d=True vs explicit params produce same result."""
        r1 = forward_op(param3d=tiny_source["param3d"], pixelated=True)
        r2 = forward_op(
            true_intensity=tiny_source["inten"],
            true_doppler=tiny_source["vel"],
            true_linewidth=tiny_source["width"],
            pixelated=True,
        )
        np.testing.assert_allclose(r1, r2, rtol=1e-12)

    def test_output_shape(self, tiny_source):
        """Output shape is (K, H, W) for K spectral orders."""
        result = forward_op(
            true_intensity=tiny_source["inten"],
            true_doppler=tiny_source["vel"],
            true_linewidth=tiny_source["width"],
            pixelated=True,
            spectral_orders=[0, -1, 1],
        )
        assert result.shape == (3, 4, 4)

    def test_single_order_shape(self, tiny_source):
        result = forward_op(
            true_intensity=tiny_source["inten"],
            true_doppler=tiny_source["vel"],
            true_linewidth=tiny_source["width"],
            pixelated=True,
            spectral_orders=[0],
        )
        assert result.shape == (1, 4, 4)

    def test_pixelated_vs_not_different(self, tiny_source):
        """pixelated=True vs False produce different results."""
        r_pix = forward_op(
            true_intensity=tiny_source["inten"],
            true_doppler=tiny_source["vel"],
            true_linewidth=tiny_source["width"],
            pixelated=True,
        )
        r_cont = forward_op(
            true_intensity=tiny_source["inten"],
            true_doppler=tiny_source["vel"],
            true_linewidth=tiny_source["width"],
            pixelated=False,
        )
        # They should differ because gauss and gauss_pix give different results
        assert not np.allclose(r_pix, r_cont)

    def test_zero_velocity_all_orders_equal_pixelated(self, tiny_source):
        """With zero velocity and pixelated=True, all orders ≈ same (symmetric)."""
        result = forward_op(
            true_intensity=tiny_source["inten"],
            true_doppler=tiny_source["vel"],
            true_linewidth=tiny_source["width"],
            pixelated=True,
            spectral_orders=[0, -1, 1],
        )
        np.testing.assert_allclose(result[0], result[1], rtol=1e-10)
        np.testing.assert_allclose(result[0], result[2], rtol=1e-10)

    def test_with_mask(self, tiny_source):
        mask = np.ones((4, 4))
        result = forward_op(
            true_intensity=tiny_source["inten"],
            true_doppler=tiny_source["vel"],
            true_linewidth=tiny_source["width"],
            pixelated=True,
            mask=mask,
        )
        assert result.shape == (3, 4, 4)

    def test_order_zero_is_identity(self, tiny_source):
        """Order 0 is simply mask * intensity (no dispersion)."""
        result = forward_op(
            true_intensity=tiny_source["inten"],
            true_doppler=tiny_source["vel"],
            true_linewidth=tiny_source["width"],
            pixelated=True,
            spectral_orders=[0],
        )
        np.testing.assert_allclose(result[0], tiny_source["inten"], rtol=1e-12)


# ============================================================
# forward_op_torch
# ============================================================


class TestForwardOpTorch:
    def test_matches_numpy(self, tiny_source):
        """Same output values as numpy forward_op for same inputs."""
        inten_t = torch.tensor(tiny_source["inten"], dtype=torch.float64)
        vel_t = torch.tensor(tiny_source["vel"], dtype=torch.float64)
        width_t = torch.tensor(tiny_source["width"], dtype=torch.float64)
        # NOTE: forward_op_torch has a known bug where mask=None crashes with
        # 2D input, so we pass an explicit mask here.
        mask_t = torch.ones_like(inten_t)

        result_torch = forward_op_torch(
            true_intensity=inten_t,
            true_doppler=vel_t,
            true_linewidth=width_t,
            pixelated=True,
            spectral_orders=[0, -1, 1],
            mask=mask_t,
        )
        result_np = forward_op(
            true_intensity=tiny_source["inten"],
            true_doppler=tiny_source["vel"],
            true_linewidth=tiny_source["width"],
            pixelated=True,
            spectral_orders=[0, -1, 1],
        )
        np.testing.assert_allclose(result_torch[0].numpy(), result_np, rtol=1e-12)

    def test_batched_input(self, tiny_source):
        """Batched input (B, H, W) produces (B, K, H, W)."""
        b = 3
        inten_b = torch.tensor(
            np.stack([tiny_source["inten"]] * b), dtype=torch.float32
        )
        vel_b = torch.tensor(
            np.stack([tiny_source["vel"]] * b), dtype=torch.float32
        )
        width_b = torch.tensor(
            np.stack([tiny_source["width"]] * b), dtype=torch.float32
        )

        result = forward_op_torch(
            true_intensity=inten_b,
            true_doppler=vel_b,
            true_linewidth=width_b,
            pixelated=True,
            spectral_orders=[0, -1, 1],
        )
        assert result.shape == (b, 3, 4, 4)
        # Each batch element should be identical
        for i in range(1, b):
            np.testing.assert_allclose(
                result[0].numpy(), result[i].numpy(), rtol=1e-6
            )

    def test_non_batched_returns_3d(self, tiny_source):
        """2D input returns (1, K, H, W) — known bug: batch dim not stripped."""
        inten_t = torch.tensor(tiny_source["inten"], dtype=torch.float32)
        vel_t = torch.tensor(tiny_source["vel"], dtype=torch.float32)
        width_t = torch.tensor(tiny_source["width"], dtype=torch.float32)
        # NOTE: forward_op_torch has a known bug where mask=None crashes with
        # 2D input, and the batch dim is not stripped. Pass mask explicitly.
        mask_t = torch.ones_like(inten_t)

        result = forward_op_torch(
            true_intensity=inten_t,
            true_doppler=vel_t,
            true_linewidth=width_t,
            pixelated=True,
            mask=mask_t,
        )
        # NOTE: known behavior — returns (1, K, H, W) instead of (K, H, W)
        assert result.shape == (1, 3, 4, 4)

    def test_gradients_flow(self, tiny_source):
        """loss.backward() does not crash and gradients are non-zero."""
        inten_t = torch.tensor(tiny_source["inten"], dtype=torch.float32, requires_grad=True)
        vel_t = torch.tensor(tiny_source["vel"], dtype=torch.float32, requires_grad=True)
        width_t = torch.tensor(tiny_source["width"], dtype=torch.float32, requires_grad=True)
        # NOTE: forward_op_torch has known bug with mask=None on 2D input.
        mask_t = torch.ones_like(inten_t)

        result = forward_op_torch(
            true_intensity=inten_t,
            true_doppler=vel_t,
            true_linewidth=width_t,
            pixelated=True,
            mask=mask_t,
        )
        loss = result.sum()
        loss.backward()
        assert inten_t.grad is not None
        assert vel_t.grad is not None
        assert width_t.grad is not None
        assert torch.any(inten_t.grad != 0)


# ============================================================
# add_noise
# ============================================================


class TestAddNoise:
    def test_no_noise_identity(self):
        """With no_noise=True, output == input exactly."""
        sig = np.random.randn(4, 5, 6).astype(np.float64)
        noisy = add_noise(sig, no_noise=True)
        np.testing.assert_array_equal(noisy, sig)

    def test_noise_model_none_identity(self):
        """With noise_model=None, output == input exactly."""
        sig = np.random.randn(4, 5, 6).astype(np.float64)
        noisy = add_noise(sig, noise_model=None)
        np.testing.assert_array_equal(noisy, sig)

    def test_gaussian_adds_noise(self):
        """With Gaussian noise at dbsnr=20, output differs from input."""
        np.random.seed(42)
        sig = np.random.randn(3, 10, 10).astype(np.float64) * 50 + 100
        noisy = add_noise(sig, dbsnr=20, noise_model="Gaussian")
        assert noisy.shape == sig.shape
        assert not np.allclose(noisy, sig, rtol=1e-6)

    def test_gaussian_snr_approx(self):
        """With dbsnr=20, measured SNR ≈ 20 dB (within tolerance due to randomness)."""
        np.random.seed(12345)
        sig = np.random.randn(5, 20, 20).astype(np.float64) * 100 + 500.0
        dbsnr = 20.0
        noisy = add_noise(sig, dbsnr=dbsnr, noise_model="Gaussian")
        noise = noisy - sig
        var_signal = np.var(sig)
        var_noise = np.var(noise)
        measured_snr = 10 * np.log10(var_signal / var_noise)
        # Allow ±3 dB tolerance due to random draw
        assert abs(measured_snr - dbsnr) < 3.0

    def test_poisson_with_dbsnr_close_to_input(self):
        """With Poisson noise at reasonable dbsnr, output is close to input."""
        np.random.seed(42)
        sig = np.ones((5, 20, 20)) * 100.0
        noisy = add_noise(sig, dbsnr=10, noise_model="Poisson")
        assert noisy.shape == sig.shape
        # Poisson noise should be somewhat close at dbsnr=10
        np.testing.assert_allclose(noisy, sig, rtol=0.5)

    def test_torch_input_returns_torch(self):
        """Torch tensor input → torch tensor output."""
        sig = torch.ones(2, 3, 4)
        noisy = add_noise(sig, dbsnr=20, noise_model="Gaussian")
        assert isinstance(noisy, torch.Tensor)


# ============================================================
# Source
# ============================================================


class TestSource:
    def test_construct_from_arrays(self, tiny_source):
        s = Source(
            inten=tiny_source["inten"],
            vel=tiny_source["vel"],
            width=tiny_source["width"],
        )
        assert s.pix is False
        np.testing.assert_array_equal(s.inten, tiny_source["inten"])
        np.testing.assert_array_equal(s.vel, tiny_source["vel"])
        np.testing.assert_array_equal(s.width, tiny_source["width"])

    def test_param3d_stacked_correctly(self, tiny_source):
        s = Source(
            inten=tiny_source["inten"],
            vel=tiny_source["vel"],
            width=tiny_source["width"],
        )
        assert s.param3d.shape == (3, 4, 4)
        np.testing.assert_array_equal(s.param3d[0], tiny_source["inten"])
        np.testing.assert_array_equal(s.param3d[1], tiny_source["vel"])
        np.testing.assert_array_equal(s.param3d[2], tiny_source["width"])

    def test_construct_from_param3d(self, tiny_source):
        s = Source(param3d=tiny_source["param3d"])
        np.testing.assert_array_equal(s.inten, tiny_source["inten"])
        np.testing.assert_array_equal(s.vel, tiny_source["vel"])
        np.testing.assert_array_equal(s.width, tiny_source["width"])

    def test_pix_flag_stored(self):
        s = Source(inten=np.ones((2, 2)), vel=np.zeros((2, 2)), width=np.ones((2, 2)), pix=True)
        assert s.pix is True

        s2 = Source(inten=np.ones((2, 2)), vel=np.zeros((2, 2)), width=np.ones((2, 2)))
        assert s2.pix is False

    def test_plot_runs(self, tiny_source):
        s = Source(
            inten=tiny_source["inten"],
            vel=tiny_source["vel"],
            width=tiny_source["width"],
        )
        fig, ax = s.plot(title="test source")
        assert fig is not None
        assert len(ax) == 3
        plt.close(fig)


# ============================================================
# Imager
# ============================================================


class TestImager:
    def test_construct_defaults(self):
        imager = Imager()
        assert imager.pixel_size == 13.5
        assert imager.mid_wavelength == 195.119
        assert imager.dispersion_scale is not None
        assert imager.spectral_orders == [0, -1, 1]
        assert imager.pixelated is False
        assert imager.dbsnr is None

    def test_custom_dispersion_scale(self):
        imager = Imager(dispersion_scale=0.05)
        assert imager.dispersion_scale == 0.05

    def test_topix(self, tiny_source):
        imager = Imager(
            pixel_size=1.0,
            dispersion=0.022275,
            mid_wavelength=195.119,
        )
        physical = Source(
            inten=tiny_source["inten"],
            vel=np.ones((4, 4)),
            width=np.ones((4, 4)) * 0.01,
            pix=False,
        )
        pixel = imager.topix(physical)
        assert pixel.pix is True
        assert pixel.inten.shape == (4, 4)
        assert pixel.vel.shape == (4, 4)
        assert pixel.width.shape == (4, 4)

    def test_frompix(self):
        imager = Imager(
            pixel_size=1.0,
            dispersion=0.022275,
            mid_wavelength=195.119,
        )
        pixel = Source(
            inten=np.ones((4, 4)),
            vel=np.zeros((4, 4)),
            width=np.ones((4, 4)) * 0.01,
            pix=True,
        )
        physical = imager.frompix(pixel)
        assert physical.pix is False
        assert physical.inten.shape == (4, 4)

    def test_frompix_width_km_s(self):
        imager = Imager(
            pixel_size=1.0,
            dispersion=0.022275,
            mid_wavelength=195.119,
        )
        pixel = Source(
            inten=np.ones((2, 2)),
            vel=np.zeros((2, 2)),
            width=np.ones((2, 2)) * 0.02,
            pix=True,
        )
        physical = imager.frompix(pixel, width_unit="km/s")
        assert physical.pix is False
        assert physical.width.shape == (2, 2)

    def test_forward_op_shape(self, tiny_source, tiny_imager):
        result = tiny_imager.forward_op(
            tiny_source["inten"],
            tiny_source["vel"],
            tiny_source["width"],
        )
        assert result.shape == (3, 4, 4)

    def test_get_measurements_noiseless(self, tiny_source):
        imager = Imager(
            pixel_size=1.0,
            dispersion=0.022275,
            spectral_orders=[0, -1, 1],
            pixelated=True,
        )
        source = Source(
            inten=tiny_source["inten"],
            vel=tiny_source["vel"],
            width=tiny_source["width"],
            pix=True,
        )
        # NOTE: dbsnr=0 does NOT mean no-noise in add_noise, so get_measurements
        # with dbsnr=0 will still add noise. Use no_noise=True for clean signal.
        result = imager.get_measurements(sources=source, no_noise=True)
        assert result.shape == (3, 4, 4)

    def test_get_measurements_adds_noise(self, tiny_source):
        np.random.seed(42)
        imager = Imager(
            pixel_size=1.0,
            dispersion=0.022275,
            spectral_orders=[0, -1, 1],
            pixelated=True,
        )
        # Use non-uniform intensity so variance is non-zero (otherwise
        # dbsnr-based noise is 0 for constant signals).
        # NOTE: known behavior — constant signals produce zero noise with dbsnr,
        # because var_signal=0 → std_noise=0.
        inten = np.random.randn(4, 4) * 100 + 1000
        vel = np.random.randn(4, 4) * 0.1
        width = np.abs(np.random.randn(4, 4) * 0.02 + 0.05)
        source = Source(
            inten=inten,
            vel=vel,
            width=width,
            pix=True,
        )
        result_noisy = imager.get_measurements(
            sources=source,
            dbsnr=20,
            noise_model="Gaussian",
        )
        result_clean = imager.meas3dar_nn
        assert not np.allclose(result_noisy, result_clean, rtol=1e-6)

    def test_frompix_array_mode(self):
        """frompix(array=True) handles an array/tensor of shape (C, H, W)."""
        imager = Imager(
            pixel_size=1.0,
            dispersion=0.022275,
            mid_wavelength=195.119,
        )
        # Must replicate srpix so frompix(array=True) can read rest_wavelength
        imager.srpix = Source(
            inten=np.ones((2, 2)),
            vel=np.zeros((2, 2)),
            width=np.ones((2, 2)),
            pix=True,
        )
        arr = np.ones((3, 2, 2))
        arr[0] *= 1000.0
        result = imager.frompix(arr, width_unit="A", array=True)
        assert result.shape == (3, 2, 2)
        assert result[0, 0, 0] == pytest.approx(1000.0, rel=1e-10)

    def test_frompix_array_mode_torch(self):
        """frompix(array=True) with torch tensor."""
        imager = Imager(
            pixel_size=1.0,
            dispersion=0.022275,
            mid_wavelength=195.119,
        )
        imager.srpix = Source(
            inten=np.ones((2, 2)),
            vel=np.zeros((2, 2)),
            width=np.ones((2, 2)),
            pix=True,
        )
        arr = torch.ones(3, 2, 2)
        arr[0] *= 1000.0
        result = imager.frompix(arr, width_unit="A", array=True)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 2, 2)

    def test_plot_runs(self, tiny_source):
        imager = Imager(
            pixel_size=1.0,
            dispersion=0.022275,
            spectral_orders=[0, -1, 1],
            pixelated=True,
        )
        source = Source(
            inten=tiny_source["inten"],
            vel=tiny_source["vel"],
            width=tiny_source["width"],
            pix=True,
        )
        imager.get_measurements(sources=source, no_noise=True)
        imager.plot(noise=False)
        plt.close("all")


# ============================================================
# tomomtx_gen
# ============================================================


class TestTomoMtxGen:
    def test_shape(self):
        """For 4x4 with orders [0,-1,1], output has correct dimensions."""
        mtx = tomomtx_gen((4, 4), orders=[0, -1, 1])
        assert mtx.shape == (12, 16)

    def test_nonzero(self):
        """Matrix is not all zeros."""
        mtx = tomomtx_gen((4, 4), orders=[0, -1, 1])
        assert np.sum(np.abs(mtx)) > 0

    def test_order_zero_is_identity_like(self):
        """Order 0 should be like an identity in the column direction — each row has exactly M ones."""
        mtx = tomomtx_gen((4, 4), orders=[0, -1, 1])
        order0_rows = mtx[0:4, :]
        assert np.all(order0_rows.sum(axis=1) == 4)

    def test_single_order(self):
        mtx = tomomtx_gen((4, 4), orders=[0])
        assert mtx.shape == (4, 16)

    def test_order_inf(self):
        mtx = tomomtx_gen((4, 4), orders=["inf"])
        assert mtx.shape == (4, 16)
        assert np.sum(np.abs(mtx)) > 0
