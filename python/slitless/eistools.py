# 2026-01-20
# Ulas Kamaci

import copy
import numpy as np
from joblib import Parallel, delayed
from eispac.extern.mpfit import mpfit
import eispac.core.fitting_functions as fit_fns
from eispac.core.scale_guess import scale_guess
from eispac.instr import calc_velocity
from scipy.interpolate import interp1d
from slitless.config import config
from slitless.forward import Source, forward_op_tomo_3d
import matplotlib.pyplot as plt
from eispac.core.eiscube import EISCube
import os, shutil, subprocess
from slitless.data_loader import (BasicDataset, param_inv_transform,
    meas_inv_transform, meas_transform, param_transform)
from torch.utils.data import DataLoader
import glob

WAVELENGTH = 195.119
DISPERSION_SCALE = 22.275

def _worker_fit_chunk(args):
    """
    Worker function for parallel fitting.
    """
    wave_row, int_row, err_row, template, parinfo, min_points = args
    n_pixels = int_row.shape[0] 
    
    n_params = len(parinfo) 
    n_gauss = template['n_gauss']
    n_poly = template['n_poly']
    min_w = template['wmin']
    max_w = template['wmax']
    oldguess = template['fit']
    
    out_params = np.zeros((n_pixels, n_params))
    out_perror = np.zeros((n_pixels, n_params))
    out_status = np.zeros(n_pixels)
    out_chi2   = np.zeros(n_pixels)

    for i in range(n_pixels):
        wave_i = wave_row[i, :]
        int_i = int_row[i, :]
        err_i = err_row[i, :]

        # Strict check matching eispac: 
        # Only fit if err > 0 AND wavelength is within window
        loc_good = np.where((err_i > 0) & (wave_i >= min_w) & (wave_i <= max_w))[0]
        
        if len(loc_good) < min_points: 
            out_status[i] = -1 # Matches eispac 'not enough data' code
            continue

        w_g = wave_i[loc_good]
        i_g = int_i[loc_good]
        e_g = err_i[loc_good]

        # Update guess
        newguess = scale_guess(w_g, i_g, oldguess, n_gauss, n_poly)
        for k in range(len(newguess)): 
            parinfo[k]['value'] = newguess[k]

        fa = {'x': w_g, 'y': i_g, 'error': e_g, 'n_gauss': n_gauss, 'n_poly': n_poly}
        
        out = mpfit(fit_fns.multigaussian_deviates, parinfo=parinfo,
                    functkw=fa, xtol=1e-6, ftol=1e-6, gtol=1e-6,
                    maxiter=2000, quiet=1)
        
        out_status[i] = out.status
        out_chi2[i] = out.fnorm / max(1, out.dof)
        
        if out.status > 0:
            out_params[i, :] = out.params
            out_perror[i, :] = out.perror

    return out_params, out_perror, out_status, out_chi2

def meas_boundary_corrector(imgr, meas4dar, param4dar):
    for i in range(param4dar.shape[0]):
        Sr = Source(
            param3d=param4dar[i],
            pix=False
        )

        intenscale = Sr.param3d[0].max()
        imgr.intenscale=intenscale
        imgr.topix(Sr)

        meas3dar = imgr.get_measurements(noise_model=None)
        corrector = np.ones_like(meas3dar)
        corrector[1:,:3] = meas3dar[1:,:3]/meas3dar[1:,[3]]
        corrector[1:,-3:] = meas3dar[1:,-3:]/meas3dar[1:,[-4]]
        meas4dar[i] *= corrector

    return meas4dar

def fit_spectra_joblib(data_cube, tmplt, n_jobs=48, component=0, fill_masked=True):
    """
    Fits spectra using Joblib, propagates errors, AND performs model-based inpainting.
    
    Args:
        fill_masked (bool): If True, replaces masked pixels in the output data 
                            with the value of the fitted model function.
    
    Returns:
        param_map, error_map, full_status, corrected_data
    """
    print(f" + Computing fit for component {component} on {n_jobs} cores...")
    
    # 1. Prepare Data & Explicit Cleaning
    # -------------------------------------------------------
    # We copy input data to safe arrays. 
    # These 'safe' arrays will eventually become the 'corrected' data.
    safe_data = np.array(data_cube.data).astype(np.float64)
    safe_errs = np.array(data_cube.uncertainty.array).astype(np.float64)
    safe_wave = np.array(data_cube.wavelength).astype(np.float64)

    # Apply Mask (Set bad data to 0 for the fitter)
    if data_cube.mask is not None and np.any(data_cube.mask):
        loc_masked = np.where(data_cube.mask == True)
        safe_data[loc_masked] = 0
        safe_errs[loc_masked] = 0
            
    # Handle "Bad Data" (negative errors) matching eispac logic
    loc_bad = np.where(safe_errs <= 0)
    safe_data[loc_bad] = 0
    safe_errs[loc_bad] = 0
    # -------------------------------------------------------

    n_rows = safe_data.shape[0]
    p_base = tmplt.parinfo
    t_base = tmplt.template
    
    # 2. Run Parallel Fit (Standard)
    # -------------------------------------------------------
    tasks = []
    for y in range(n_rows):
        tasks.append((
            safe_wave[y, :, :], 
            safe_data[y, :, :], 
            safe_errs[y, :, :], 
            t_base, 
            copy.deepcopy(p_base), 
            7 # min_points
        ))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_worker_fit_chunk)(t) for t in tasks
    )
    
    full_params = np.stack([r[0] for r in results])
    full_perror = np.stack([r[1] for r in results]) 
    full_status = np.stack([r[2] for r in results])

    # 3. Parameter Extraction (Same as before)
    # -------------------------------------------------------
    idx_peak = component * 3
    idx_cent = component * 3 + 1
    idx_width = component * 3 + 2

    raw_peak = full_params[:, :, idx_peak]
    raw_cent = full_params[:, :, idx_cent]
    raw_width = full_params[:, :, idx_width]
    err_peak = full_perror[:, :, idx_peak]
    err_cent = full_perror[:, :, idx_cent]
    err_width = full_perror[:, :, idx_width]

    intensity = np.sqrt(2 * np.pi) * raw_peak * raw_width
    
    with np.errstate(divide='ignore', invalid='ignore'):
        err_intensity = intensity * np.sqrt((err_peak/raw_peak)**2 + (err_width/raw_width)**2)
    err_intensity = np.nan_to_num(err_intensity)

    rest_wave = p_base[idx_cent]['value'] 
    # Calculate absolute velocity to ensure 1-to-1 matching with datacube spectra
    velocity = 299792.458 * (raw_cent - rest_wave) / rest_wave
    err_velocity = 299792.458 * (err_cent / rest_wave)

    # Filter bad fits
    bad_mask = (full_status <= 0)
    intensity[bad_mask] = 0
    velocity[bad_mask] = 0
    raw_width[bad_mask] = 0
    err_intensity[bad_mask] = 0
    err_velocity[bad_mask] = 0

    param_map = np.stack((intensity, velocity, raw_width))
    error_map = np.stack((err_intensity, err_velocity, err_width))

    # 4. Inpainting / Correction Logic
    # -------------------------------------------------------
    corrected_data = safe_data # This is our output array (masked values are currently 0)
    
    if fill_masked and data_cube.mask is not None:
        print(" + Inpainting masked pixels with fitted model values...")
        
        # We need the model function to reconstruct the spectrum
        # eispac.core.fitting_functions.multigaussian(x, params, n_gauss, n_poly)
        n_gauss = t_base['n_gauss']
        n_poly = t_base['n_poly']
        
        # Find pixels (y, x) that have at least one masked wavelength point
        # This optimization prevents iterating over millions of clean pixels
        mask_3d = data_cube.mask
        bad_y, bad_x = np.where(np.any(mask_3d, axis=2))
        unique_bad_pixels = list(set(zip(bad_y, bad_x)))

        for y, x in unique_bad_pixels:
            # We can ONLY inpaint if the fit actually worked!
            if full_status[y, x] > 0:
                # 1. Get the wavelengths for this pixel
                w_pixel = safe_wave[y, x, :]
                
                # 2. Get the fitted parameters for this pixel
                p_pixel = full_params[y, x, :]
                
                # 3. Calculate the full model curve
                # CORRECTION: p_pixel (params) comes FIRST, w_pixel (x) comes SECOND
                model_spectrum = fit_fns.multigaussian(p_pixel, w_pixel, n_gauss, n_poly)
                
                # 4. Identify which specific points were masked
                pixel_mask = mask_3d[y, x, :]
                
                # 5. Overwrite ONLY the masked points with the model values
                corrected_data[y, x, pixel_mask] = model_spectrum[pixel_mask]

    return param_map, error_map, full_status, corrected_data


def eis_to_ssi_interpolator(data_cube, cube_wavelength=None, lamdim=21, kind='cubic', pad_mode='edge'):
    """
    Function to convert EIS data cube into SSI data cube via interpolation.
    
    Args:
        data_cube (eispac_cube): eispac datacube object holding the data and
        attributes of the EIS observation
        lamdim (int): desired number of pixels in the wavelength dimension of
        the SSI cube
    Returns:
        data_cube_i (ndarray): interpolated array (lamdim, Ny, Nx)
    """
    if isinstance(data_cube, EISCube):
        cube_wavelength = data_cube.wavelength
        cube_data = data_cube.data
    else:
        cube_data = data_cube

    wavelength_grid_target = (
        WAVELENGTH + 
        DISPERSION_SCALE/1000 * (np.arange(lamdim)-lamdim//2)
    )

    wavelength_grid_source = cube_wavelength

    # Prepare output array with new wavelength dimension
    y_sz, x_sz, _ = cube_data.shape
    lamdim = wavelength_grid_target.shape[0]
    data_cube_interpolated = np.zeros((y_sz, x_sz, lamdim), dtype=cube_data.dtype)

    for iy in range(y_sz):
        for ix in range(x_sz):
            wl_src = wavelength_grid_source[iy, ix, :]
            flux_src = cube_data[iy, ix, :]

            # Check for monotonicity & valid values
            valid = np.isfinite(wl_src) & np.isfinite(flux_src)
            if valid.sum() < 2:
                data_cube_interpolated[iy, ix, :] = np.nan
                continue

            try:
                if pad_mode == 'edge':
                    edge_vals = (flux_src[valid][0], flux_src[valid][-1])
                else:
                    edge_vals = 0
                interp_fn = interp1d(wl_src[valid], flux_src[valid], kind=kind,
                                        bounds_error=False, fill_value=edge_vals)
                data_cube_interpolated[iy, ix, :] = interp_fn(wavelength_grid_target)
            except Exception as e:
                data_cube_interpolated[iy, ix, :] = np.nan
    
    return data_cube_interpolated

def eis_random_cropper(param3d, data_cube, numcopies=5, patchsize=(64,64)):
    """
    Picks a pool of random patches, scores them by total intensity sum, 
    and returns the top 'numcopies' patches.

    Args:
        param3d (np.ndarray): Array of shape (3, H, W).
        data_cube (np.ndarray): Array of shape (H, W, L).
        numcopies (int): Number of top patches to return.
        patchsize (tuple): (height, width) of the patch.

    Returns:
        param3d_cropped (list): List of top param patches.
        data_cube_cropped (list): List of top data patches.
    """
    assert param3d.shape[1:] == data_cube.shape[:2], "Shapes don't match!"
    H, W = param3d.shape[1:]
    patchH, patchW = patchsize
    
    # 1. Determine pool size based on your logic
    # We ensure we have enough candidates, but cap the effort at 32
    pool_size = min(32, 3 * numcopies)
    
    # 2. Generate random top-left coordinates
    # High is exclusive in randint, so we add 1 to include the last valid index
    max_y = H - patchH
    max_x = W - patchW
    
    # Safety check if image is smaller than patch
    if max_y < 0 or max_x < 0:
        raise ValueError(f"Image size ({H},{W}) is smaller than patch size {patchsize}")

    rand_y = np.random.randint(0, max_y + 1, size=pool_size)
    rand_x = np.random.randint(0, max_x + 1, size=pool_size)

    candidates = []

    # 3. Extract and Score Candidates
    for y, x in zip(rand_y, rand_x):
        # Extract patches (using views temporarily for speed)
        p_patch_view = param3d[:, y:y+patchH, x:x+patchW]
        d_patch_view = data_cube[y:y+patchH, x:x+patchW]
        
        # Score: Total Summed Intensity of the first parameter map
        # using p_patch_view[0] assumes param3d[0] is the intensity map
        score = np.sum(p_patch_view[0])
        
        # Store tuple: (score, y, x)
        candidates.append({
            'score': score,
            'p_patch': p_patch_view.copy(), # Save actual copy now
            'd_patch': d_patch_view.copy(),  # Save actual copy now
            'patch_coords': (y, x)
        })

    # 4. Sort by score (descending) and take top numcopies
    # Lambda key sorts by the 'score' value in the dictionary
    candidates.sort(key=lambda item: item['score'], reverse=True)
    
    top_candidates = candidates[:numcopies]

    # 5. Unpack into lists
    param3d_cropped = [c['p_patch'] for c in top_candidates]
    data_cube_cropped = [c['d_patch'] for c in top_candidates]
    patch_coords = [c['patch_coords'] for c in top_candidates]

    return param3d_cropped, data_cube_cropped, patch_coords

def eis_random_cropper2(param3d, data_cube, meas, mask, numcopies=5, patchsize=(64,64)):
    """
    Picks a pool of random patches avoiding mask=1 areas by pre-calculating 
    valid top-left coordinates using an Integral Image (Summed Area Table).

    Args:
        param3d (np.ndarray): Array of shape (3, H, W).
        data_cube (np.ndarray): Array of shape (H, W, L).
        meas (np.ndarray): Array of shape (K, H, W).
        mask (np.ndarray): Binary array of shape (H, W). 1 = Bad, 0 = Good.
        numcopies (int): Number of top patches to return.
        patchsize (tuple): (height, width) of the patch.
    """
    # 0. Shape Validations
    assert param3d.shape[1:] == data_cube.shape[:2], "Param and Data shapes don't match!"
    assert param3d.shape[1:] == meas.shape[1:], "Data and Meas shapes don't match!"
    assert mask.shape == param3d.shape[1:], "Mask shape does not match H, W!"

    H, W = param3d.shape[1:]
    pH, pW = patchsize
    
    # The last valid top-left index
    max_y = H - pH
    max_x = W - pW

    if max_y < 0 or max_x < 0:
        raise ValueError(f"Image ({H},{W}) smaller than patch {patchsize}")

    # 1. Compute 'Validity Map' using Integral Image
    # We want to know the sum of mask pixels for every possible patch window.
    # Integral image allows calculating sum of any rect in O(1).
    
    # Pad mask with one row/col of zeros for easy indexing (A, B, C, D trick)
    # integral_img[y, x] = sum of all pixels above-left of (y,x)
    pad_mask = np.pad(mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    integral_img = pad_mask.cumsum(axis=0).cumsum(axis=1)

    # We want the sum inside the box for top-left (y,x).
    # The box corners in the integral image are:
    # Top-Left: (y, x)      Bottom-Right: (y+pH, x+pW)
    # Note: Because of padding, integral_img index [y] corresponds to image coord [y-1].
    # So image coord 0 is integral index 1. 
    # The sum of box [y:y+pH, x:x+pW] is:
    # I[y+pH, x+pW] - I[y, x+pW] - I[y+pH, x] + I[y, x]
    
    # We define ranges for the "Top-Left" of the patches we want to check
    # We check all y from 0 to max_y, and x from 0 to max_x
    
    # Vectorized calculation for all valid top-left positions:
    # S[y,x] is the sum of mask pixels for a patch starting at y,x
    patch_sums = (integral_img[pH : max_y + pH + 1, pW : max_x + pW + 1] 
                  - integral_img[0 : max_y + 1,      pW : max_x + pW + 1] 
                  - integral_img[pH : max_y + pH + 1, 0 : max_x + 1] 
                  + integral_img[0 : max_y + 1,      0 : max_x + 1])

    # 2. Find Valid Coordinates (where sum is 0)
    valid_y, valid_x = np.nonzero(patch_sums == 0)
    
    num_valid = len(valid_y)
    if num_valid == 0:
        raise ValueError("Mask is too dense; no valid patch locations found.")

    # 3. Sample from Valid Coordinates
    pool_size = min(32, 3 * numcopies)
    
    if num_valid <= pool_size:
        # If valid spots are scarce, take them all
        chosen_indices = np.arange(num_valid)
    else:
        # Randomly sample indices without replacement
        chosen_indices = np.random.choice(num_valid, size=pool_size, replace=False)

    candidates = []

    for idx in chosen_indices:
        y = valid_y[idx]
        x = valid_x[idx]

        # Extract views
        p_patch = param3d[:, y:y+pH, x:x+pW]
        d_patch = data_cube[y:y+pH, x:x+pW]
        m_patch = meas[:, y:y+pH, x:x+pW]

        score = np.sum(p_patch[0]) # Score by intensity
        
        candidates.append({
            'score': score,
            'p_patch': p_patch.copy(),
            'd_patch': d_patch.copy(),
            'm_patch': m_patch.copy(),
            'patch_coords': (y, x)
        })

    # 4. Sort and Return
    candidates.sort(key=lambda item: item['score'], reverse=True)
    top_candidates = candidates[:numcopies]

    param3d_cropped = [c['p_patch'] for c in top_candidates]
    data_cube_cropped = [c['d_patch'] for c in top_candidates]
    meas_cropped = [c['m_patch'] for c in top_candidates]
    patch_coords = [c['patch_coords'] for c in top_candidates]

    return param3d_cropped, data_cube_cropped, meas_cropped, patch_coords

def quickplot(path=None, array=None, patch_coords=None, patchsize=(64,64), title=None):
    p1,p2=patchsize
    if path is not None:
        data = np.load(path, allow_pickle=True).item()
    elif array is not None:
        data = {
            'int': array[0],
            'vel': array[1],
            'width': array[2]
        }
    sr = Source(param3d=np.stack([data['int'], data['vel'], data['width']]), pix=False)
    tit = path[-20:] if path is not None else title
    fig, ax = sr.plot(title=tit)
    if patch_coords is not None:
        for y, x in patch_coords:
            print(f'({x},{y})')
            ax[0].add_patch(plt.Rectangle((x, y), p2, p1, fill=False, edgecolor='blue', linewidth=2))
    plt.show()
    return fig, ax

def quickplot6(meas, param3d, show=False):
    fig, ax = plt.subplots(2,3,figsize=(15,10))
    im=ax[0,0].imshow(meas[0], cmap='hot')
    ax[0,0].set_title('Meas 0')
    fig.colorbar(im, ax=ax[0,0])
    im=ax[0,1].imshow(meas[1], cmap='hot')
    ax[0,1].set_title('Meas -1')
    fig.colorbar(im, ax=ax[0,1])
    im=ax[0,2].imshow(meas[2], cmap='hot')
    ax[0,2].set_title('Meas +1')
    fig.colorbar(im, ax=ax[0,2])
    im=ax[1,0].imshow(param3d[0], cmap='hot')
    ax[1,0].set_title('True Intensity')
    fig.colorbar(im, ax=ax[1,0])
    im=ax[1,1].imshow(param3d[1], cmap='seismic')
    ax[1,1].set_title('True Velocity')
    fig.colorbar(im, ax=ax[1,1])
    im=ax[1,2].imshow(param3d[2], cmap='plasma')
    ax[1,2].set_title('True Linewidth')
    fig.colorbar(im, ax=ax[1,2])
    if show:
        plt.show()
    return fig, ax

def example_figurer(
    pathdir=str(config.data_root / 'eis_data' / 'datasets' / 'dset_v4'),
    fold='train'
):
    files = glob.glob(os.path.join(pathdir, 'data', fold, 'data*.npy'))
    np.random.shuffle(files)
    
    savedir = os.path.join(pathdir, 'smallset', fold, 'examples/')
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
        
    num_samples = min(32, len(files))
    for i in range(num_samples):
        data_i = np.load(files[i], allow_pickle=True).item()
        
        meas_ = np.stack([data_i['meas_0'], data_i['meas_-1'], data_i['meas_1']])
        param3d_ = np.stack([data_i['int'], data_i['vel'], data_i['width']])
        
        fig, ax = quickplot6(meas_, param3d_)
        fig.savefig(savedir+f'data_{i:02}.png', dpi=300)
        np.save(os.path.join(savedir, f'data_{i:02}.npy'), data_i)
        plt.close()

def small_train_generator(
    inds, 
    pathdir=str(config.data_root / 'eis_data' / 'datasets' / 'dset_v4' / 'smallset' / 'train'),
):
    param3d_l, meas_l, meas2_l, cube_l = [], [], [], []
    for i in inds:
        data_i = np.load(os.path.join(pathdir, 'examples', f'data_{i:02}.npy'), allow_pickle=True).item()
        meas_l.append(np.stack([data_i['meas_0'], data_i['meas_-1'], data_i['meas_1'], data_i['meas_-2'], data_i['meas_2']]))
        param3d_l.append(np.stack([data_i['int'], data_i['vel'], data_i['width']]))
        cube_l.append(data_i['datacube'])
        meas2_l.append(forward_op_tomo_3d(data_i['datacube'].transpose(2,0,1), orders=[0,-1,1,-2,2]))
    data = {
        'meas': np.array(meas_l),
        'meas_damped': np.array(meas2_l),
        'param3d': np.array(param3d_l),
        'datacube': np.array(cube_l),
        'example_inds': inds,
    }
    np.save(os.path.join(pathdir, 'eis_train_5_dsetv4.npy'), data)

def small_test_generator(
    inds, 
    pathdir=str(config.data_root / 'eis_data' / 'datasets' / 'dset_v4' / 'smallset' / 'test'),
):
    param3d_l, meas_l, meas2_l = [], [], []
    for i in inds:
        data_i = np.load(os.path.join(pathdir, 'examples', f'data_{i:02}.npy'), allow_pickle=True).item()
        meas_l.append(np.stack([data_i['meas_0'], data_i['meas_-1'], data_i['meas_1'], data_i['meas_-2'], data_i['meas_2']]))
        param3d_l.append(np.stack([data_i['int'], data_i['vel'], data_i['width']]))
        meas2_l.append(forward_op_tomo_3d(data_i['datacube'].transpose(2,0,1), orders=[0,-1,1,-2,2]))
    data = {
        'meas': np.array(meas_l),
        'meas_damped': np.array(meas2_l),
        'param3d': np.array(param3d_l),
        'example_inds': inds,
    }
    np.save(os.path.join(pathdir, 'eis_test_5_dsetv4.npy'), data)
    
def download_eis(date_str, local_dir):
    """
    Downloads EIS HDF5 data from the MSSL mirror (UK), which is currently online.
    """
    # 1. Setup paths
    os.makedirs(local_dir, exist_ok=True)
    
    # Parse date: '20120216' -> '2012', '02', '16'
    year, month, day = date_str[0:4], date_str[4:6], date_str[6:8]
    
    # NEW URL: MSSL Mirror
    base_url = f"https://vsolar.mssl.ucl.ac.uk/eispac/hdf5/{year}/{month}/{day}/"
    
    filenames = [f"eis_{date_str}.data.h5", f"eis_{date_str}.head.h5"]
    
    # Check for curl (preferred) or wget
    has_curl = shutil.which("curl") is not None

    for fname in filenames:
        local_path = os.path.join(local_dir, fname)
        remote_url = base_url + fname

        if os.path.exists(local_path):
            print(f"File exists, skipping: {fname}")
            continue

        print(f"Downloading {fname} from MSSL...")
        
        if has_curl:
            # -L follows redirects, -o writes to file
            cmd = ["curl", "-L", "-o", local_path, remote_url]
        else:
            # -O (uppercase) writes to file in wget
            cmd = ["wget", "-O", local_path, remote_url]

        try:
            subprocess.run(cmd, check=True)
            print(f"Success: {fname}")
        except subprocess.CalledProcessError:
            print(f"FAILED to download {fname}.")
            # Clean up empty/failed file
            if os.path.exists(local_path):
                os.remove(local_path)