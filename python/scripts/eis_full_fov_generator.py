# Ulas Kamaci
# Script to generate full FOV EIS datasets, skipping the random cropping phase.
# Simulates measurements for orders 0, -1, 1 and generates summary figures.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback
import eispac, os
import numpy as np
from tqdm import tqdm
from slitless.forward import forward_op_tomo_3d
from slitless.eistools import (eis_to_ssi_interpolator,
    fit_spectra_joblib, download_eis)
from slitless.config import config

pathdir = str(config.data_root / 'eis_data')
savedir = str(config.data_root / 'eis_data/datasets/full_fov')

# Ensure output directories exist
os.makedirs(os.path.join(savedir, 'data'), exist_ok=True)
os.makedirs(os.path.join(savedir, 'figs'), exist_ok=True)

WAVELENGTH = 195.119
DISP_SCALE = 13.5 * 1.65 / 1000

dates = [
   '20070124_181113',
   '20070720_111614',
   '20070127_034020',
   '20070128_061012',
   '20070326_183342',
   '20070512_094534',
   '20070702_120742',
   '20070708_105820',
   '20070929_140227',
   '20071004_140220',
   '20071211_002416',
   '20080106_105243',
   '20080203_073210',
]

def plot_full_fov(param3d, meas, date, savedir):
    """
    Generates a 2x3 subplot figure displaying the ground truth parameters
    and the resulting simulated measurements.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle(f'Full FOV Scan: {date}', fontsize=16)
    
    # Row 1: Parameters
    im0 = axs[0, 0].imshow(param3d[0], cmap='hot')
    axs[0, 0].set_title('True Intensity')
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Using a typical solar velocity range for color centering
    im1 = axs[0, 1].imshow(param3d[1], cmap='seismic', vmin=-30, vmax=30)
    axs[0, 1].set_title('True Velocity (km/s)')
    plt.colorbar(im1, ax=axs[0, 1])
    
    im2 = axs[0, 2].imshow(param3d[2], cmap='plasma')
    axs[0, 2].set_title('True Linewidth (Å)')
    plt.colorbar(im2, ax=axs[0, 2])
    
    # Row 2: Measurements
    im3 = axs[1, 0].imshow(meas[0], cmap='hot')
    axs[1, 0].set_title('Measurement (Order 0)')
    plt.colorbar(im3, ax=axs[1, 0])
    
    im4 = axs[1, 1].imshow(meas[1], cmap='hot')
    axs[1, 1].set_title('Measurement (Order -1)')
    plt.colorbar(im4, ax=axs[1, 1])
    
    im5 = axs[1, 2].imshow(meas[2], cmap='hot')
    axs[1, 2].set_title('Measurement (Order +1)')
    plt.colorbar(im5, ax=axs[1, 2])
    
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, 'figs', f'summary_{date}.png'), dpi=200)
    plt.close(fig)

if __name__ == '__main__':
    for num, date in enumerate(tqdm(dates)):
        try:
            print('\n{}/{}: {}'.format(num+1, len(dates), date))
            download_eis(date, pathdir + 'l2/')
            eis_filepath = pathdir + f'l2/eis_{date}.data.h5'
            template_filepath = pathdir + 'templates/fe_12_195_119.2c.template.h5'

            data_cube = eispac.read_cube(eis_filepath, window=WAVELENGTH)
            tmplt = eispac.read_template(template_filepath)

            os.remove(eis_filepath)
            os.remove(pathdir + f'l2/eis_{date}.head.h5')

            print(f"Computing fit for {date} (Joblib)...")
            param3d, err3d, status2d, cube_cor = fit_spectra_joblib(
                data_cube, tmplt, n_jobs=48, component=0)
                
            data_cube_i = eis_to_ssi_interpolator(cube_cor, data_cube.wavelength, lamdim=21)
            data_cube_i *= DISP_SCALE

            meas = forward_op_tomo_3d(data_cube_i.transpose(2,0,1), orders=[0, -1, 1])
            
            # Generate and save the 6-panel summary plot
            plot_full_fov(param3d, meas, date, savedir)

            out = {
                'meas_0': meas[0], 'meas_-1': meas[1], 'meas_1': meas[2],
                'int': param3d[0], 'vel': param3d[1], 'width': param3d[2],
                'datacube': data_cube_i, 'filename': f'{date}'
            }
            np.save(os.path.join(savedir, 'data', f'data_{date}_full.npy'), out)
        except Exception as e:
            traceback.print_exc()
            continue
