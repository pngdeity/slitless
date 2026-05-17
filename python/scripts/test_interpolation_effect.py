import os
import numpy as np
import matplotlib.pyplot as plt
import eispac
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame

from slitless.eistools import download_eis, fit_spectra_joblib, eis_to_ssi_interpolator
from slitless.recon import smart_fit_spectra_joblib
from slitless.config import config

# Constants
DISP_SCALE_A = 0.022275  
WAVELENGTH = 195.119     
SPEED_OF_LIGHT = 299792.458 

def get_metrics(true, est):
    rmse = np.sqrt(np.mean((true - est) ** 2, axis=(-1, -2)))
    bias = np.mean(est - true, axis=(-1, -2))
    return rmse, bias

def main():
    date = '20070124_181113'
    pathdir = str(config.data_root / 'eis_data')
    eis_filepath = os.path.join(pathdir, f'l2/eis_{date}.data.h5')
    template_filepath = os.path.join(pathdir, 'templates/fe_12_195_119.2c.template.h5')

    if not os.path.exists(eis_filepath):
        print(f"Downloading {date}...")
        download_eis(date, os.path.join(pathdir, 'l2/'))

    print("Reading EIS cube...")
    data_cube = eispac.read_cube(eis_filepath, window=WAVELENGTH)
    tmplt = eispac.read_template(template_filepath)

    print("Cropping a 64x64 region for testing...")
    y_center = data_cube.data.shape[0] // 2
    x_center = data_cube.data.shape[1] // 2
    coords_bl = data_cube.wcs.array_index_to_world(y_center - 32, x_center - 32, 0)
    coords_tr = data_cube.wcs.array_index_to_world(y_center + 31, x_center + 31, 0)
    
    eis_frame = wcs_to_celestial_frame(data_cube.wcs)
    lower_left = [None, SkyCoord(Tx=coords_bl[1].Tx, Ty=coords_bl[1].Ty, frame=eis_frame)]
    upper_right = [None, SkyCoord(Tx=coords_tr[1].Tx, Ty=coords_tr[1].Ty, frame=eis_frame)]
    raster_cutout = data_cube.crop(lower_left, upper_right)

    print("\n[1] Fitting Raw Data (Ground Truth) ...")
    param_raw, _, _, cube_cor = fit_spectra_joblib(raster_cutout, tmplt, n_jobs=-1, component=0)
    
    print("\n[2] Interpolating to SSI Grid (Zero-Padded vs Edge-Padded) ...")
    cube_zero = eis_to_ssi_interpolator(cube_cor, raster_cutout.wavelength, lamdim=21, pad_mode='zero')
    cube_edge = eis_to_ssi_interpolator(cube_cor, raster_cutout.wavelength, lamdim=21, pad_mode='edge')

    lamdim = 21
    wave_ssi = WAVELENGTH + DISP_SCALE_A * (np.arange(lamdim) - lamdim // 2)

    print("\n[3] Fitting Interpolated Cubes ...")
    print("  -> Fitting Zero-Padded Cube...")
    param_zero = smart_fit_spectra_joblib(cube_zero.transpose(2,0,1), tmplt, wave_ssi, n_jobs=-1, component=0)
    
    print("  -> Fitting Edge-Padded Cube...")
    param_edge = smart_fit_spectra_joblib(cube_edge.transpose(2,0,1), tmplt, wave_ssi, n_jobs=-1, component=0)

    # Convert widths to km/s
    param_raw[2] *= SPEED_OF_LIGHT / WAVELENGTH
    param_zero[2] *= SPEED_OF_LIGHT / WAVELENGTH
    param_edge[2] *= SPEED_OF_LIGHT / WAVELENGTH

    rmse_zero, bias_zero = get_metrics(param_raw, param_zero)
    rmse_edge, bias_edge = get_metrics(param_raw, param_edge)

    print("\n" + "=" * 60)
    print("          INTERPOLATION EVALUATION SUMMARY")
    print("=" * 60)
    print("Method: Zero-Padding (Old)")
    print(f" Int RMSE={rmse_zero[0]:8.2f}, Bias={bias_zero[0]:8.2f}")
    print(f" Vel RMSE={rmse_zero[1]:8.3f}, Bias={bias_zero[1]:8.3f} km/s")
    print(f" Wid RMSE={rmse_zero[2]:8.3f}, Bias={bias_zero[2]:8.3f} km/s")
    print("-" * 60)
    print("Method: Edge-Padding (New)")
    print(f" Int RMSE={rmse_edge[0]:8.2f}, Bias={bias_edge[0]:8.2f}")
    print(f" Vel RMSE={rmse_edge[1]:8.3f}, Bias={bias_edge[1]:8.3f} km/s")
    print(f" Wid RMSE={rmse_edge[2]:8.3f}, Bias={bias_edge[2]:8.3f} km/s")
    print("=" * 60)

    # Plot 1D spectrum for brightest pixel to visually compare tails
    y_max, x_max = np.unravel_index(np.argmax(param_raw[0]), param_raw[0].shape)
    
    plt.figure(figsize=(10, 6))
    plt.plot(raster_cutout.wavelength[y_max, x_max], cube_cor[y_max, x_max], 'ko', label='Raw (Inpainted)', markersize=5)
    plt.plot(wave_ssi, cube_zero[y_max, x_max], 'b.-', label='Zero-Padded SSI', linewidth=1.5)
    plt.plot(wave_ssi, cube_edge[y_max, x_max], 'r.-', label='Edge-Padded SSI', linewidth=1.5)
    plt.title(f'1D Spectrum Comparison at Brightest Pixel (y={y_max}, x={x_max})')
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Flux Density')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('interpolation_comparison_1d.png', dpi=200, bbox_inches='tight')
    print("\nSaved 1D spectrum comparison to 'interpolation_comparison_1d.png'")

    print("\n--- DEBUG INFO ---")
    raw_wave = raster_cutout.wavelength[y_max, x_max]
    print(f"Raw Wavelengths ({len(raw_wave)} points):\n{raw_wave}")
    print(f"SSI Wavelengths ({len(wave_ssi)} points):\n{wave_ssi}")

    wmin = tmplt.template['wmin']
    wmax = tmplt.template['wmax']
    print(f"\nTemplate Boundaries: wmin = {wmin:.4f} Å, wmax = {wmax:.4f} Å")
    
    raw_kept = (raw_wave >= wmin) & (raw_wave <= wmax)
    ssi_kept = (wave_ssi >= wmin) & (wave_ssi <= wmax)
    
    print(f"\nRaw points kept by template: {raw_kept.sum()} out of {len(raw_wave)}")
    print(f"SSI points kept by template: {ssi_kept.sum()} out of {len(wave_ssi)}")
    
    print("\nSSI points outside template bounds (IGNORED BY FITTER):")
    for w in wave_ssi[~ssi_kept]:
        print(f"  {w:.4f} Å")

if __name__ == '__main__':
    main()