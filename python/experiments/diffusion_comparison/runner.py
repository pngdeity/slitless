import os
import gc
import torch
import numpy as np
import datetime
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from slitless.forward import Imager
from slitless.recon import nn_solver, dps_solver, conddiff_solver, Reconstructor_Multi

# ==============================================================================
# 1. RUN CONFIGURATION
# ==============================================================================
DATA_FILE = 'eis_test_50_dsetv6.npy'
PATH_DATA = '/home/kamo/resources/slitless/data/datasets/baseline/'
OUTPUT_DIR = '/home/kamo/resources/slitless/python/experiments/diffusion_comparison/output/'

# (dbsnr, noise_model)  —  None/None means noiseless
CONFIGS = [
    (None, None),
    (20,   'gaussian'),
    (30,   'gaussian'),
]

K = 3
SPECTRAL_ORDERS = [0, -1, 1]

# ==============================================================================
# 2. METHOD-SPECIFIC PATHS
# ==============================================================================
UNET_MODEL_PATHS = {
    (None, None):       '2026_05_28__00_09_08_diffusion_unet_NF_64_BS_32_LR_0.0002_EP_50_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_100_None_K_3_dset_v6_logzscale',
    (20,  'gaussian'):  '2026_05_31__13_07_15_diffusion_unet_NF_64_BS_32_LR_0.0002_EP_100_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_20_gaussian_K_3_dset_v6_logzscale',
    (30,  'gaussian'):  '2026_05_31__12_00_32_diffusion_unet_NF_64_BS_32_LR_0.0002_EP_100_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_30_gaussian_K_3_dset_v6_logzscale',
}

# DPS uses the single unconditional model for all noise levels
DPS_MODEL_PATH = 'model-10.pt'  # inside run_all_lr_1e-4_cosine_b32_logz/ (hardcoded in dps_solver)

CONDDIFF_RUN_NAMES = {
    (None, None):       'run_all_lr1e-4_cosine_b32_conditional_logz',
    (20,  'gaussian'):  '2026_05_31__04_12_50_all_lr_1e-4_cosine_b32_numdetectors_3_global_logz_conditional_Gaussian_20',
    (30,  'gaussian'):  '2026_06_02__21_03_55_all_lr_1e-4_cosine_b32_numdetectors_3_global_logz_conditional_Gaussian_30',
}

# ==============================================================================
# 3. RUNNER
# ==============================================================================
def run_all():
    print(f'Loading dataset: {DATA_FILE}...')
    data = np.load(os.path.join(PATH_DATA, DATA_FILE), allow_pickle=True).item()
    param4dar = data['param3d']           # (N, 3, 64, 64)  physical units
    meas_all  = data['meas'][:, :K]      # (N, K, 64, 64)  noiseless

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'\nResults will be saved to: {OUTPUT_DIR}\n')

    results_summary = {}
    global_start = time.time()

    for cfg_idx, (dbsnr, noise_model) in enumerate(CONFIGS):
        snr_str  = f'{dbsnr} dB {noise_model}' if dbsnr is not None else 'Noiseless'
        cfg_name = f'dbsnr_{dbsnr}_{noise_model}'
        print('\n' + '='*70)
        print(f' CONFIG {cfg_idx+1}/{len(CONFIGS)}: {snr_str}')
        print('='*70)

        Imgr = Imager(
            pixelated=True,
            spectral_orders=SPECTRAL_ORDERS,
            dispersion_scale=0.022275,
            mid_wavelength=195.119,
            dbsnr=dbsnr,
            noise_model=noise_model,
        )

        # ----- U-Net --------------------------------------------------------
        print('\n  [1/3] U-Net')
        Rec_unet = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            meas4dar=meas_all,
            pix=False,
            solver=nn_solver,
            intenscaling=False,
            model_path=UNET_MODEL_PATHS[(dbsnr, noise_model)],
        )
        t0 = time.time()
        Rec_unet.solve(num_realizations=1)
        print(f'    done in {time.time()-t0:.1f}s')
        _print_metrics(Rec_unet, '    ')
        gc.collect(); torch.cuda.empty_cache()

        pkl_path = os.path.join(OUTPUT_DIR, f'Rec_unet_{cfg_name}.pickle')
        with open(pkl_path, 'wb') as f:
            pickle.dump(Rec_unet, f)
        print(f'    saved -> {pkl_path}')

        # ----- DPS ----------------------------------------------------------
        print('\n  [2/3] DPS (unconditional diffusion)')
        Rec_dps = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            meas4dar=meas_all,
            pix=False,
            solver=dps_solver,
            intenscaling=False,
            model_path=DPS_MODEL_PATH,
            grad_scale=[0.5, 0.5, 0.5],
            num_samples=10,
        )
        t0 = time.time()
        Rec_dps.solve(num_realizations=1)
        print(f'    done in {time.time()-t0:.1f}s')
        _print_metrics(Rec_dps, '    ')
        gc.collect(); torch.cuda.empty_cache()

        pkl_path = os.path.join(OUTPUT_DIR, f'Rec_dps_{cfg_name}.pickle')
        with open(pkl_path, 'wb') as f:
            pickle.dump(Rec_dps, f)
        print(f'    saved -> {pkl_path}')

        # ----- CondDiff -----------------------------------------------------
        print('\n  [3/3] CondDiff (conditional diffusion)')
        Rec_cond = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            meas4dar=meas_all,
            pix=False,
            solver=conddiff_solver,
            intenscaling=False,
            run_name=CONDDIFF_RUN_NAMES[(dbsnr, noise_model)],
            model_path='model-10.pt',
            num_samples=10,
        )
        t0 = time.time()
        Rec_cond.solve(num_realizations=1)
        print(f'    done in {time.time()-t0:.1f}s')
        _print_metrics(Rec_cond, '    ')
        gc.collect(); torch.cuda.empty_cache()

        pkl_path = os.path.join(OUTPUT_DIR, f'Rec_cond_{cfg_name}.pickle')
        with open(pkl_path, 'wb') as f:
            pickle.dump(Rec_cond, f)
        print(f'    saved -> {pkl_path}')

        results_summary[(dbsnr, noise_model)] = {
            'unet': _extract_metrics(Rec_unet),
            'dps':  _extract_metrics(Rec_dps),
            'cond': _extract_metrics(Rec_cond),
        }

        elapsed = time.time() - global_start
        avg_t   = elapsed / (cfg_idx + 1)
        eta     = avg_t * (len(CONFIGS) - cfg_idx - 1)
        print(f'\n  [Elapsed: {datetime.timedelta(seconds=int(elapsed))} | ETA: {datetime.timedelta(seconds=int(eta))}]')

    # ---------- summary pickle ----------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, 'results_summary.pickle')
    with open(summary_path, 'wb') as f:
        pickle.dump(results_summary, f)
    print(f'\nSaved results summary -> {summary_path}')

    _print_latex_table(results_summary)

    total = time.time() - global_start
    print(f'\nAll configs done in {datetime.timedelta(seconds=int(total))}.')


def _extract_metrics(Rec):
    rmse = Rec.rmse_phy.mean(axis=(0, 1))
    bias = Rec.bias_phy.mean(axis=(0, 1))
    return {'rmse': rmse, 'bias': bias, 'time': Rec.times.mean()}


def _print_metrics(Rec, prefix=''):
    rmse = Rec.rmse_phy.mean(axis=(0, 1))
    bias = Rec.bias_phy.mean(axis=(0, 1))
    print(f'{prefix}Int  RMSE={rmse[0]:.2f}  Bias={bias[0]:.2f}')
    print(f'{prefix}Vel  RMSE={rmse[1]:.3f}  Bias={bias[1]:.3f} km/s')
    print(f'{prefix}Wid  RMSE={rmse[2]:.3f}  Bias={bias[2]:.3f} km/s')


def _print_latex_table(results_summary):
    methods = [('unet', 'U-Net'), ('dps', 'DPS'), ('cond', 'CondDiff')]
    snr_labels = {(None, None): r'$\infty$', (20, 'gaussian'): '20', (30, 'gaussian'): '30'}

    print('\n\n' + '='*70)
    print(' LATEX TABLE (SNR sweep, K=3, Gaussian noise)')
    print('='*70)
    for cfg in CONFIGS:
        res = results_summary.get(cfg)
        if not res:
            continue
        label = snr_labels[cfg]
        for key, name in methods:
            m = res[key]
            rmse, bias = m['rmse'], m['bias']
            print(
                f'\\multirow{{3}}{{*}}{{{label}}} & {name} & '
                f'{rmse[0]:.1f} & {bias[0]:.1f} & '
                f'{rmse[1]:.3f} & {bias[1]:.3f} & '
                f'{rmse[2]:.3f} & {bias[2]:.3f} \\\\'
            )


if __name__ == '__main__':
    run_all()
