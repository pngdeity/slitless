# Ulas Kamaci
# 2025-12-26
# This code tests if forward_op_tomo_3d works properly

from slitless.forward import Source, Imager, forward_op_tomo_3d, datacube_generator
from slitless.plotting import uiuc_im
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset
from slitless.config import config

path_data = str(config.data_root / 'datasets/baseline')
data = 'eis_5_64x64.npy' # 64x64 EIS images

param3dar = np.load(path_data+data)[1]
# param4dar = uiuc_im()

orders = [0,-1,1,2,-2]
Imgr = Imager(pixelated=True, spectral_orders=orders)

sr = Source(param3d=param3dar,pix=True)
meas_gauss = Imgr.get_measurements(sources=sr)

r = datacube_generator(param3dar, lamdim=21, pixelated=True)

meas_tomo = forward_op_tomo_3d(r, orders=orders)

print(f'meas_gauss shape:{meas_gauss.shape}')
print(f'meas_tomo shape:{meas_tomo.shape}')

fig, axs = plt.subplots(len(orders), 3, figsize=(18, 6 * len(orders)))

for i, order in enumerate(orders):
    if len(orders) == 1:
        ax_row = axs
    else:
        ax_row = axs[i]
    vmin = np.min([meas_gauss[i], meas_tomo[i]])
    vmax = np.max([meas_gauss[i], meas_tomo[i]])

    im0 = ax_row[0].imshow(meas_gauss[i], cmap='hot', vmin=vmin, vmax=vmax)
    ax_row[0].set_title('meas_gauss (Order {})'.format(order))
    plt.colorbar(im0, ax=ax_row[0])

    im1 = ax_row[1].imshow(meas_tomo[i], cmap='hot', vmin=vmin, vmax=vmax)
    ax_row[1].set_title('meas_tomo (Order {})'.format(order))
    plt.colorbar(im1, ax=ax_row[1])

    diff = meas_gauss[i] - meas_tomo[i]
    im2 = ax_row[2].imshow(diff, cmap='hot')
    ax_row[2].set_title('Difference (gauss - tomo) (Order {})'.format(order))
    plt.colorbar(im2, ax=ax_row[2])

    for ax in ax_row:
        ax.axis('off')

plt.tight_layout()
plt.show()