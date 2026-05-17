import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
import eispac
from slitless.eistools import download_eis
from slitless.config import config


def plot_fov_spectra(data_filepath, center_wavelength_nm=19.51, window_width_nm=10.0, 
                     num_positions=9, positions=None):
    """
    Plot spectra at multiple locations in the FOV within a specified wavelength window.
    
    Parameters:
    -----------
    data_filepath : str
        Path to the EIS data file (.h5)
    center_wavelength_nm : float
        Center wavelength in nanometers (default: 19.51 nm)
    window_width_nm : float
        Width of the wavelength window in nanometers (default: 10.0 nm)
    num_positions : int
        Number of positions to plot (default: 9, arranged in a 3x3 grid)
    positions : list of tuples, optional
        List of (iy, ix) indices for specific positions. If None, uses a grid.
    """
    # Convert nm to Angstroms (1 nm = 10 Å)
    center_wavelength_angstrom = center_wavelength_nm * 10.0
    window_width_angstrom = window_width_nm * 10.0
    
    print(f"Loading EIS data cube around {center_wavelength_nm} nm ({center_wavelength_angstrom:.1f} Å)...")
    print(f"Window width: {window_width_nm} nm ({window_width_angstrom:.1f} Å)")
    
    # Read the data cube for the specified wavelength window
    data_cube = eispac.read_cube(data_filepath, window=center_wavelength_angstrom)
    
    print(f"\nData cube shape: {data_cube.data.shape}")
    print(f"  - Spatial dimensions (y, x): {data_cube.data.shape[0]} x {data_cube.data.shape[1]}")
    print(f"  - Spectral dimension: {data_cube.data.shape[2]}")
    
    # Determine positions to plot
    if positions is None:
        # Create a grid of positions
        ny, nx = data_cube.data.shape[0], data_cube.data.shape[1]
        # Calculate grid spacing
        grid_size = int(np.sqrt(num_positions))
        if grid_size * grid_size != num_positions:
            grid_size = int(np.ceil(np.sqrt(num_positions)))
        
        y_indices = np.linspace(ny // 4, 3 * ny // 4, grid_size, dtype=int)
        x_indices = np.linspace(nx // 4, 3 * nx // 4, grid_size, dtype=int)
        
        positions = []
        for iy in y_indices:
            for ix in x_indices:
                if len(positions) < num_positions:
                    positions.append((iy, ix))
    else:
        num_positions = len(positions)
    
    print(f"\nPlotting spectra at {num_positions} positions:")
    
    # Extract spectra at each position
    spectra_data = []
    positions_info = []
    
    for i, (iy, ix) in enumerate(positions):
        # Get world coordinates
        coords = data_cube.wcs.array_index_to_world(iy, ix, 0)
        x_arcsec = coords[1].Tx.value
        y_arcsec = coords[1].Ty.value
        
        # Extract spectrum
        wavelength = data_cube.wavelength[iy, ix, :]
        intensity = data_cube.data[iy, ix, :]
        uncertainty = data_cube.uncertainty.array[iy, ix, :]
        
        # Filter to the wavelength window
        wvl_min = center_wavelength_angstrom - window_width_angstrom / 2
        wvl_max = center_wavelength_angstrom + window_width_angstrom / 2
        mask = (wavelength >= wvl_min) & (wavelength <= wvl_max)
        
        spectra_data.append({
            'wavelength': wavelength[mask],
            'intensity': intensity[mask],
            'uncertainty': uncertainty[mask]
        })
        
        positions_info.append({
            'iy': iy,
            'ix': ix,
            'x_arcsec': x_arcsec,
            'y_arcsec': y_arcsec
        })
        
        print(f"  Position {i+1}: (iy={iy}, ix={ix}) -> (Tx={x_arcsec:.1f}\", Ty={y_arcsec:.1f}\")")
    
    # Create the plot
    grid_size = int(np.ceil(np.sqrt(num_positions)))
    fig = plt.figure(figsize=(15, 15))
    plot_grid = fig.add_gridspec(nrows=grid_size, ncols=grid_size, hspace=0.4, wspace=0.3)
    
    for i, (spec_data, pos_info) in enumerate(zip(spectra_data, positions_info)):
        row = i // grid_size
        col = i % grid_size
        ax = fig.add_subplot(plot_grid[row, col])
        
        # Remove NaN values
        valid_mask = ~np.isnan(spec_data['intensity'])
        wavelengths = spec_data['wavelength'][valid_mask]
        intensities = spec_data['intensity'][valid_mask]
        uncertainties = spec_data['uncertainty'][valid_mask]
        
        # Plot with error bars
        ax.errorbar(wavelengths, intensities, yerr=uncertainties,
                   ls='-', marker='o', markersize=2, linewidth=1, capsize=2)
        
        ax.set_title(f"Pos ({pos_info['ix']}, {pos_info['iy']})\n"
                    f"Tx={pos_info['x_arcsec']:.1f}\", Ty={pos_info['y_arcsec']:.1f}\"",
                    fontsize=9)
        ax.set_xlabel('Wavelength [Å]', fontsize=8)
        ax.set_ylabel('Intensity', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at center wavelength
        ax.axvline(center_wavelength_angstrom, color='r', linestyle='--', 
                  linewidth=1, alpha=0.5, label=f'{center_wavelength_nm} nm')
        ax.legend(fontsize=7, loc='best')
    
    # Remove empty subplots
    for i in range(num_positions, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        fig.delaxes(plot_grid[row, col].subplot_spec)
    
    plt.suptitle(f'EIS Spectra at Multiple FOV Locations\n'
                f'Window: {center_wavelength_nm:.2f} nm ± {window_width_nm/2:.2f} nm '
                f'({center_wavelength_angstrom:.1f} Å ± {window_width_angstrom/2:.1f} Å)',
                y=0.98, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
    return fig, spectra_data, positions_info


if __name__ == '__main__':
    # Default data file path (same as in eis_fitting.py)
    date = '20070124_181113'
    data_dir = str(config.data_root / 'eis_data/l2')
    data_filepath = os.path.join(data_dir, f'eis_{date}.data.h5')
    
    if not os.path.exists(data_filepath):
        print(f"Data file not found. Downloading {date} from MSSL mirror...")
        download_eis(date, data_dir)
    
    # Plot spectra at multiple positions
    # You can customize:
    # - center_wavelength_nm: center of the window (default: 19.51 nm)
    # - window_width_nm: width of the window (default: 10.0 nm)
    # - num_positions: number of positions to plot (default: 9, in a 3x3 grid)
    # - positions: specific (iy, ix) positions as a list of tuples
    
    fig, spectra_data, positions_info = plot_fov_spectra(
        data_filepath,
        center_wavelength_nm=19.51,
        window_width_nm=10.0,
        num_positions=9
    )
    
    # Example: Plot at specific positions
    # fig, spectra_data, positions_info = plot_fov_spectra(
    #     data_filepath,
    #     center_wavelength_nm=19.51,
    #     window_width_nm=10.0,
    #     positions=[(50, 50), (100, 100), (150, 150), (200, 200)]
    # )
