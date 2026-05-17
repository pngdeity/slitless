import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
import eispac
from slitless.config import config


if __name__ == '__main__':
    # Read in the fit template and EIS observation
    date = '20070124_181113'
    pathdir = str(config.data_root / 'eis_data')
    eispac.db.download_hdf5_data(
        # filename=f'eis_l0_{date}.fits.gz', 
        filename=f'eis_{date}', 
        local_top=pathdir+'l2/'
    )
    data_filepath = str(config.data_root / 'eis_data/l2/eis_20070124_181113.data.h5')
    template_filepath = str(config.data_root / 'eis_data/templates/fe_12_195_119.2c.template.h5')
    tmplt = eispac.read_template(template_filepath)
    data_cube = eispac.read_cube(data_filepath, tmplt.central_wave)

    # Print information about the data cube dimensions and coordinates
    print("\nData cube shape:", data_cube.data.shape)
    print("\nCoordinate ranges:")
    # Check corners and center
    y_indices = [0, data_cube.data.shape[0]//2, data_cube.data.shape[0]-1]
    x_indices = [0, data_cube.data.shape[1]//2, data_cube.data.shape[1]-1]
    for iy in y_indices:
        for ix in x_indices:
            coords = data_cube.wcs.array_index_to_world(iy, ix, 0)
            print(f"Position (y={iy}, x={ix}): Tx={coords[1].Tx.value:.1f}, Ty={coords[1].Ty.value:.1f} arcsec")

    # Select a cutout of the raster
    eis_frame = wcs_to_celestial_frame(data_cube.wcs)
    lower_left = [None, SkyCoord(Tx=700, Ty=-150, unit=u.arcsec, frame=eis_frame)]
    upper_right = [None, SkyCoord(Tx=850, Ty=0, unit=u.arcsec, frame=eis_frame)]
    raster_cutout = data_cube.crop(lower_left, upper_right)

    # Calculate the extent for plotting
    cutout_extent = [lower_left[1].Tx.value, upper_right[1].Tx.value,
                    lower_left[1].Ty.value, upper_right[1].Ty.value]

    # Fit the data and save it to disk
    fit_res = eispac.fit_spectra(raster_cutout, tmplt, ncpu='max', unsafe_mp=True)
    save_filepaths = eispac.save_fit(fit_res, save_dir='cwd')

    # Find indices and world coordinates of max intensity
    sum_data_inten = raster_cutout.sum_spectra().data
    iy, ix = np.unravel_index(sum_data_inten.argmax(), sum_data_inten.shape)
    ex_world_coords = raster_cutout.wcs.array_index_to_world(iy, ix, 0)[1]
    y_arcsec, x_arcsec = ex_world_coords.Ty.value, ex_world_coords.Tx.value

    # Print diagnostic information
    print("\nDiagnostic Information:")
    print(f"Coordinates in arcsec: x={x_arcsec:.1f}, y={y_arcsec:.1f}")

    # Find the corresponding position in the original data cube using the arcsec coordinates
    all_coords = data_cube.wcs.array_index_to_world(np.arange(data_cube.data.shape[0])[:,None], 
                                                  np.arange(data_cube.data.shape[1])[None,:], 
                                                  np.zeros((data_cube.data.shape[0], data_cube.data.shape[1])))
    tx_grid = all_coords[1].Tx.value
    ty_grid = all_coords[1].Ty.value
    dist = np.sqrt((tx_grid - x_arcsec)**2 + (ty_grid - y_arcsec)**2)
    orig_iy, orig_ix = np.unravel_index(dist.argmin(), dist.shape)

    print(f"\nFound corresponding position in original cube:")
    orig_coords = data_cube.wcs.array_index_to_world(orig_iy, orig_ix, 0)[1]
    print(f"Original indices: iy={orig_iy}, ix={orig_ix}")
    print(f"Original coordinates: x={orig_coords.Tx.value:.1f}, y={orig_coords.Ty.value:.1f}")

    print("\nCutout data at position:")
    data_x = raster_cutout.wavelength[iy, ix, :]
    data_y = raster_cutout.data[iy, ix, :]
    data_err = raster_cutout.uncertainty.array[iy, ix, :]
    print(f"Wavelength range: {data_x.min():.2f} to {data_x.max():.2f}")
    print(f"Intensity range: {data_y.min():.2f} to {data_y.max():.2f}")

    print("\nOriginal data at corresponding position:")
    full_spectrum = data_cube.data[orig_iy, orig_ix, :]
    full_wavelength = data_cube.wavelength[orig_iy, orig_ix, :]
    full_uncertainty = data_cube.uncertainty.array[orig_iy, orig_ix, :]
    print(f"Wavelength range: {full_wavelength.min():.2f} to {full_wavelength.max():.2f}")
    print(f"Intensity range: {full_spectrum.min():.2f} to {full_spectrum.max():.2f}")

    # Load all available windows for this position
    print("\nLoading all spectral windows...")
    all_windows = []
    all_data = []
    all_wavelengths = []
    all_uncertainties = []

    wininfo = eispac.read_wininfo(data_filepath)
    for window in wininfo:
        print(f"\nLoading {window['line_id']} window ({window['wvl_min']:.2f} - {window['wvl_max']:.2f} Å)")
        cube = eispac.read_cube(data_filepath, window=(window['wvl_min'] + window['wvl_max'])/2)
        all_windows.append(window)
        all_data.append(cube.data[orig_iy, orig_ix, :])
        all_wavelengths.append(cube.wavelength[orig_iy, orig_ix, :])
        all_uncertainties.append(np.abs(cube.uncertainty.array[orig_iy, orig_ix, :]))

    # Create a multi-panel figure for all spectral windows
    fig = plt.figure(figsize=(15, 15))
    n_rows = (len(all_windows) + 3) // 4  # 4 windows per row
    plot_grid = fig.add_gridspec(nrows=n_rows, ncols=4, hspace=0.4, wspace=0.3)

    for i in range(len(all_windows)):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(plot_grid[row, col])
        
        # Validate data before plotting
        valid_mask = ~np.isnan(all_data[i])  # Remove NaN values if any
        wavelengths = all_wavelengths[i][valid_mask]
        intensities = all_data[i][valid_mask]
        uncertainties = all_uncertainties[i][valid_mask]
        
        ax.errorbar(wavelengths, intensities, yerr=uncertainties,
                    ls='', marker='o', markersize=3)
        ax.set_title(f"{all_windows[i]['line_id']}\n{wavelengths.mean():.1f} Å", fontsize=10)
        ax.grid(True, alpha=0.3)
        if row == n_rows-1:  # Only add xlabel for bottom row
            ax.set_xlabel('Wavelength [$\AA$]')
        if col == 0:  # Only add ylabel for leftmost column
            ax.set_ylabel('Intensity')

    # Remove any empty subplots
    for i in range(len(all_windows), n_rows * 4):
        row = i // 4
        col = i % 4
        fig.delaxes(plot_grid[row, col].subplot_spec)

    plt.suptitle(f'All EIS Spectral Windows at position (x={x_arcsec:.1f}", y={y_arcsec:.1f}")', y=0.95)
    plt.show()

    # Extract data profile and interpolate fit at higher spectral resolution
    fit_x, fit_y = fit_res.get_fit_profile(coords=[iy,ix], num_wavelengths=100)
    c0_x, c0_y = fit_res.get_fit_profile(0, coords=[iy,ix], num_wavelengths=100)
    c1_x, c1_y = fit_res.get_fit_profile(1, coords=[iy,ix], num_wavelengths=100)
    c2_x, c2_y = fit_res.get_fit_profile(2, coords=[iy,ix], num_wavelengths=100)

    # Make a multi-panel figure with the cutout and example profiles
    fig = plt.figure(figsize=[15,5])
    plot_grid = fig.add_gridspec(nrows=1, ncols=3, wspace=0.3)

    # Plot the intensity map
    data_subplt = fig.add_subplot(plot_grid[0,0])
    data_subplt.imshow(sum_data_inten, origin='lower', extent=cutout_extent)
    data_subplt.scatter(x_arcsec, y_arcsec, color='r', marker='x')
    data_subplt.set_title('Data Cutout\n'+raster_cutout.meta['mod_index']['date_obs'])
    data_subplt.set_xlabel('Solar-X [arcsec]')
    data_subplt.set_ylabel('Solar-Y [arcsec]')

    # Plot the fitted components
    profile_subplt = fig.add_subplot(plot_grid[0,1])
    profile_subplt.errorbar(data_x, data_y, yerr=data_err, ls='', marker='o', color='k')
    profile_subplt.plot(fit_x, fit_y, color='b', label='Combined profile')
    profile_subplt.plot(c0_x, c0_y, color='r', label=fit_res.fit['line_ids'][0])
    profile_subplt.plot(c1_x, c1_y, color='r', ls='--', label=fit_res.fit['line_ids'][1])
    profile_subplt.plot(c2_x, c2_y, color='g', label='Background')
    profile_subplt.set_title(f'Fitted Components at (iy={iy}, ix={ix})')
    profile_subplt.set_xlabel('Wavelength [$\AA$]')
    profile_subplt.set_ylabel('Intensity ['+raster_cutout.unit.to_string()+']')
    profile_subplt.legend(loc='upper left', frameon=False)

    # Let's modify the third panel to show both spectra for comparison
    full_subplt = fig.add_subplot(plot_grid[0,2])
    full_subplt.errorbar(data_x, data_y, yerr=data_err, 
                        ls='', marker='o', color='k', label='Cutout Data')
    full_subplt.errorbar(full_wavelength, full_spectrum, yerr=full_uncertainty, 
                        ls='', marker='s', color='r', label='Original Data', alpha=0.5)
    full_subplt.set_title(f'Spectrum Comparison at (iy={iy}, ix={ix})')
    full_subplt.set_xlabel('Wavelength [$\AA$]')
    full_subplt.set_ylabel('Intensity ['+data_cube.unit.to_string()+']')
    full_subplt.legend(loc='upper left', frameon=False)

    plt.show()