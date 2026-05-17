# 2026-01-20
# Ulas Kamaci
# Post-APJ revision EIS fitting & interpolation routine for EIS dataset generation
# With additional correction for masked/bad data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback
import eispac, os
import numpy as np
from tqdm import tqdm
from slitless.forward import forward_op_tomo_3d, Source
from slitless.eistools import (eis_to_ssi_interpolator,
    eis_random_cropper2, fit_spectra_joblib, quickplot, download_eis)
from slitless.config import config

pathdir = str(config.data_root / 'eis_data')
savedir = str(config.data_root / 'eis_data/datasets/dset_v4')
WAVELENGTH = 195.119
DISP_SCALE = 13.5*1.65/1000

# dates = [
# #    '20070124_181113',
# #    '20120317_190944',
# #    '20120216_182857',
# #    '20070720_111614',
# #    '20070127_034020',
# #    '20070128_061012',
# #    '20070326_183342',
# #    '20070512_094534',
# #    '20070702_120742',
# #    '20070708_105820',
# #    '20070929_140227',
# #    '20071004_140220',
# #    '20071211_002416',
# #    '20080106_105243',
# #    '20080203_073210',
# #    '20091211_195014',
#    '20090114_233904',
# #    '20150807_022045'
# ]

#example_line: 'https://sdc.uio.no/vol/fits/eis/level0/2006/12/01/eis_l0_20061201_151432.fits.gz\n'
with open(savedir+'/meta/file_names.txt') as f:
    dates = [line.split('/')[-1].split('.')[0][7:] for line in f.read().splitlines()]

dates = dates[1756:]
if __name__ == '__main__':
    # NB: The "name guard" above is important for running safe, parallel fitting
    #     in a stand-alone script. If running in an interactive shell, eispac will
    #     default to a single core. If you _really_ know what you are doing,
    #     you can override it. Just be careful.
    for num, date in enumerate(tqdm(dates)):
        try:
            print('{}/{}: {}'.format(num+1,len(dates), date))

            # download file if it doesn't already exist
            # eispac.db.download_hdf5_data(
            #     # filename=f'eis_l0_{date}.fits.gz', 
            #     filename=f'eis_{date}', 
            #     local_top=pathdir+'l2/'
            # )
            download_eis(date, pathdir + 'l2/')

            # Select local files (relative paths are fine)
            eis_filepath = pathdir + f'l2/eis_{date}.data.h5'
            template_filepath = pathdir + 'templates/fe_12_195_119.2c.template.h5'

            # Load the data and fit template
            # Note: read_cube() performs basic pointing corrections and
            #       applies the pre-flight radiometric calibration
            data_cube = eispac.read_cube(eis_filepath, window=195.119)
            tmplt = eispac.read_template(template_filepath)

            if min(data_cube.data.shape[:2]) < 64:
                print(f'{date} FOV dimensions {data_cube.data.shape[:2]} too small (<64), skipping')
                continue

            # remove the downloaded data and head files and the fit file
            os.remove(eis_filepath)
            os.remove(pathdir + f'l2/eis_{date}.head.h5')

            # Fit the spectra
            fit_path = str(config.data_root / f'eis_data/fits/eis_{date}.fe_12_195_119.2c-0.fit.h5')
        # 3. FIT OR LOAD FIT
            # if os.path.exists(fit_path):
            #     print(f"Loading cached fit for {date}...")
            #     fit_res = eispac.read_fit(fit_path)
            #     param3d = np.stack(( 
            #         fit_res.get_map(0, 'int').data, 
            #         fit_res.get_map(0, 'vel').data, 
            #         fit_res.get_map(0, 'width').data
            #     ))
            # else:
            print(f"Computing fit for {date} (Joblib)...")
            tmplt = eispac.read_template(template_filepath)
            param3d, err3d, status2d, cube_cor = fit_spectra_joblib(
                data_cube, tmplt, n_jobs=48, component=0)
                
            # Interpolate the data-cube into SSI detector coordinates
            data_cube_i = eis_to_ssi_interpolator(cube_cor, data_cube.wavelength, lamdim=21)
            data_cube_i *= DISP_SCALE

            # Feed the data-cubes through SSI forward model
            meas = forward_op_tomo_3d(data_cube_i.transpose(2,0,1), orders=[0,-1,1,-2,2])

            #ADD ADDITIONAL MASK HERE (int<100, vel>68.5 etc.) 
            val_mask = (
                (param3d[0] < 100.0) | 
                (param3d[0] > 25000.0) | 
                (np.abs(param3d[1]) > 68.5) | 
                (param3d[2] < 0.022)
            )
            mask = val_mask | (err3d[0]==0)

            param3d_c, dc_c, meas_c, patch_coords = eis_random_cropper2(
                param3d=param3d, data_cube=data_cube_i, meas=meas,
                mask=mask, numcopies=15, patchsize=(64,64))
            fig, ax = quickplot(array=param3d, patch_coords=patch_coords, 
                patchsize=(64,64), title=f'{date} Full')

            for i in range(len(dc_c)):
                out = {
                    'meas_0': meas_c[i][0],
                    'meas_-1': meas_c[i][1],
                    'meas_1': meas_c[i][2],
                    'meas_-2': meas_c[i][3],
                    'meas_2': meas_c[i][4],
                    'int': param3d_c[i][0],
                    'vel': param3d_c[i][1],
                    'width': param3d_c[i][2],
                    'datacube': dc_c[i],
                    'filename': f'{date}',
                    'patch_coords': patch_coords[i]
                }
                np.save(savedir+f'data/data_{date}_{i}.npy', out)
        except Exception as e:
            # Print the exception message
            traceback.print_exc()
            continue