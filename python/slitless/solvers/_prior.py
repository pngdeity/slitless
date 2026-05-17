"""Prior-based solver for slitless spectral imaging."""

import numpy as np

from slitless.config import config


def prior_solver(
    meas=None,
    imager=None,
    frac1=0.8620,
    cent1=-0.95 * (195.11723 / 299792.458) + 195.11723,
    wid1=1.28 * 0.022275,
    **kwargs,
):
    if imager is not None:
        meas = imager.meas3dar.copy()

    int0 = meas[0].copy()

    if imager is not None:
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale

    vel1_pix_0 = (cent1 - mid_wave) / disp_scale
    wid1_pix_0 = wid1 / disp_scale

    recon_int = int0 * frac1
    recon_vel = vel1_pix_0 * np.ones_like(int0)
    recon_wid = wid1_pix_0 * np.ones_like(int0)

    recon = np.stack((recon_int, recon_vel, recon_wid), axis=0)

    return recon, []
