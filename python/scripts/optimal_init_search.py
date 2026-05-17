import numpy as np
import eispac
from eispac.core.fitting_functions import multigaussian_deviates, multigaussian
from eispac.core.scale_guess import scale_guess
from eispac.extern.mpfit import mpfit
from copy import deepcopy
import matplotlib.pyplot as plt
from slitless.config import config

def main():
    datasets = [
        'eis_train_5_dsetv5.npy',
        'eis_train_50_dsetv5.npy',
        'eis_train_1000_dsetv5.npy'
    ]

    template_filepath = str(config.data_root / 'eis_data/templates/fe_12_195_119.2c.template.h5')
    tmplt = eispac.read_template(template_filepath)
    DISP_SCALE_A = 0.022275
    WAVELENGTH_CENTER = 195.119

    for dset_file in datasets:
        print(f"\n\n{'='*60}")
        print(f" ANALYZING DATASET: {dset_file}")
        print(f"{'='*60}")
        data_path = str(config.data_root / f'datasets/baseline/{dset_file}')
        data = np.load(data_path, allow_pickle=True).item()
        
        cubes = data['datacube']
        
        print("Calculating Global Mean Spectrum...")
        if cubes.ndim == 4:
            mean_spec_pix = np.mean(cubes, axis=(0,1,2))
        elif cubes.ndim == 3:
            mean_spec_pix = np.mean(cubes, axis=(0,1))
            
        lamdim = len(mean_spec_pix)
        
        # Convert pixel intensity to physical flux density for the fitter
        mean_spec_phy = mean_spec_pix / DISP_SCALE_A
        wave = WAVELENGTH_CENTER + DISP_SCALE_A * (np.arange(lamdim) - lamdim // 2)
        errs = np.sqrt(np.clip(mean_spec_phy, 20.0, None))
        
        print("Fitting 2-Gauss + Bkg Template to Global Mean...")
        
        p_base = deepcopy(tmplt.parinfo)
        t_base = tmplt.template
        n_gauss = t_base['n_gauss']
        n_poly = t_base['n_poly']
        
        guess = scale_guess(wave, mean_spec_phy, t_base['fit'], n_gauss, n_poly)
        for k in range(len(guess)):
            p_base[k]['value'] = guess[k]
            
        functkw = {'x': wave, 'y': mean_spec_phy, 'error': errs, 'n_gauss': n_gauss, 'n_poly': n_poly}
        fit = mpfit(multigaussian_deviates, parinfo=p_base, functkw=functkw, quiet=1)
        
        params = fit.params
        peak1, cent1, wid1 = params[0], params[1], params[2]
        peak2, cent2, wid2 = params[3], params[4], params[5]
        
        area1 = np.sqrt(2 * np.pi) * peak1 * wid1
        area2 = np.sqrt(2 * np.pi) * peak2 * wid2
        
        bg_params = params.copy()
        bg_params[:3*n_gauss] = 0.0
        bg_shape = multigaussian(bg_params, wave, n_gauss, n_poly)
        bg_sum = np.sum(bg_shape) * DISP_SCALE_A
        
        total_flux = area1 + area2 + bg_sum
        frac1 = area1 / total_flux
        frac2 = area2 / total_flux
        frac_bg = bg_sum / total_flux
        
        bg_shape_norm = bg_shape / max(np.sum(bg_shape), 1e-12)
        
        print("\n" + "-"*60)
        print(f" OPTIMAL INITIALIZATIONS ({dset_file})")
        print("-"*60)
        print(f"frac1 = {frac1:.4f}, frac2 = {frac2:.4f}, frac_bg = {frac_bg:.4f},")
        print(f"cent1 = {cent1:.5f}, wid1 = {wid1:.5f},")
        print(f"cent2 = {cent2:.5f}, wid2 = {wid2:.5f},")
        print("bg_shape_norm = [")
        print("    " + ", ".join([f"{val:.5f}" for val in bg_shape_norm]))
        print("]")
    
    # Calculate default template values for comparison
    lamdim = 21
    wave = WAVELENGTH_CENTER + DISP_SCALE_A * (np.arange(lamdim) - lamdim // 2)
    params_t = np.array([p['value'] for p in tmplt.parinfo])
    peak1_t, cent1_t, wid1_t = params_t[0], params_t[1], params_t[2]
    peak2_t, cent2_t, wid2_t = params_t[3], params_t[4], params_t[5]
    
    area1_t = np.sqrt(2 * np.pi) * peak1_t * wid1_t
    area2_t = np.sqrt(2 * np.pi) * peak2_t * wid2_t
    
    bg_params_t = params_t.copy()
    bg_params_t[:3*n_gauss] = 0.0
    n_gauss = tmplt.template['n_gauss']
    n_poly = tmplt.template['n_poly']
    bg_shape_t = multigaussian(bg_params_t, wave, n_gauss, n_poly)
    bg_sum_t = np.sum(bg_shape_t) * DISP_SCALE_A
    
    total_flux_t = area1_t + area2_t + bg_sum_t
    frac1_t = area1_t / total_flux_t
    frac2_t = area2_t / total_flux_t
    frac_bg_t = bg_sum_t / total_flux_t
    
    print("\n" + "="*60)
    print(" DEFAULT TEMPLATE VALUES (For Comparison)")
    print("="*60)
    print(f"frac1 = {frac1_t:.4f}, frac2 = {frac2_t:.4f}, frac_bg = {frac_bg_t:.4f}")
    print(f"cent1 = {cent1_t:.5f}, wid1 = {wid1_t:.5f}")
    print(f"cent2 = {cent2_t:.5f}, wid2 = {wid2_t:.5f}")
    print("="*60)
    
if __name__ == '__main__':
    main()