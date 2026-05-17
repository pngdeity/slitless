import numpy as np
import os
from slitless.forward import forward_op, Source, Imager

HERE = os.path.dirname(__file__)


def test_forward_op_reproduces_baseline():
    intensity = np.load(os.path.join(HERE, 'intensity.npy'))
    velocity = np.load(os.path.join(HERE, 'velocity.npy'))
    linewidth = np.load(os.path.join(HERE, 'linewidth.npy'))
    expected = np.load(os.path.join(HERE, 'meas.npy'))

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
