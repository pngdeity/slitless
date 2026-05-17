import numpy as np
import os
from slitless.forward import forward_op, Source, Imager

outdir = os.path.dirname(__file__)

rng = np.random.RandomState(42)

intensity = rng.randint(50, 1000, size=(8, 8)).astype(np.float64)
velocity = rng.rand(8, 8).astype(np.float64) * 2 - 1
linewidth = rng.rand(8, 8).astype(np.float64) * 0.5 + 0.1

param3d = np.stack([intensity, velocity, linewidth])

source = Source(inten=intensity, vel=velocity, width=linewidth, param3d=param3d)
imager = Imager(
    pixel_size=1,
    dispersion=0.022275,
    spectral_orders=[0, -1, 1],
    pixelated=True,
    dbsnr=0,
    noise_model='poisson',
)
meas = imager.forward_op(intensity, velocity, linewidth)

np.save(os.path.join(outdir, 'param3d.npy'), param3d)
np.save(os.path.join(outdir, 'intensity.npy'), intensity)
np.save(os.path.join(outdir, 'velocity.npy'), velocity)
np.save(os.path.join(outdir, 'linewidth.npy'), linewidth)
np.save(os.path.join(outdir, 'meas.npy'), meas)
print("Golden inputs saved.")
print(f"  param3d:   {param3d.shape}")
print(f"  intensity: {intensity.shape}")
print(f"  velocity:  {velocity.shape}")
print(f"  linewidth: {linewidth.shape}")
print(f"  meas:      {meas.shape}")
