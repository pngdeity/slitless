"""Shared utilities for solver functions."""

from slitless.config import config


def _extract_meas(imager):
    """Extract a copy of the measurement array from an Imager instance."""
    return imager.meas3dar.copy()


def _extract_mask(imager):
    """Extract a copy of the spatial mask from an Imager instance."""
    return imager.mask.copy()


def _extract_wavelength_params(imager):
    """Extract optical wavelength and dispersion parameters.

    Returns:
        tuple: (mid_wavelength, dispersion_scale)
    """
    if imager is not None:
        return imager.mid_wavelength, imager.dispersion_scale
    return config.mid_wavelength, config.dispersion_scale


def _extract_rest_wavelength(imager):
    """Extract rest wavelength from imager or config fallback."""
    if imager is not None and hasattr(imager, 'srpix'):
        return imager.srpix.rest_wavelength
    return config.wavelength
