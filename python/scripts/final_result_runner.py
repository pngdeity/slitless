import os
import numpy as np
import datetime
import pickle
import time
import matplotlib
matplotlib.use('Agg') # Prevents GUI crashes when generating hundreds of plots
import matplotlib.pyplot as plt
from slitless.forward import Imager
from slitless.recon import scipy_solver_parallel2, smart2, smart2_twostage, nn_solver, Reconstructor_Multi
from slitless.config import config

# ==============================================================================
# 1. RUN CONFIGURATION
# ==============================================================================
# Choose the method to evaluate: '1dmap', 'mart', or 'unet'
METHOD = 'mart'

# The test dataset to use
DATA_FILE = 'eis_test_100_dsetv5.npy'
PATH_DATA = str(config.data_root / 'datasets/baseline')

# The 7 configurations to run: (K, dbsnr). None for dbsnr means Noiseless (gamma=inf).
CONFIGS = [
    (2, None),
    (3, None),
    (4, None),
    (5, None),
    (3, 10),
    (3, 20),
    (3, 30)
]

# ==============================================================================
# 2. METHOD SPECIFIC HYPERPARAMETERS
# ==============================================================================
# Update these dictionaries with the optimal values you found!

# --- 1D MAP (scipy_solver_parallel) ---
# NOTE: 1D MAP doesn't use the background components, so it just needs lam_v and lam_w
OPTIMAL_PARAMS_1DMAP = {
    (2, None): {'lam_v': 7.1e4, 'lam_w': 1.3e5},
    (3, None): {'lam_v': 5.1e5, 'lam_w': 1.0e6},
    (4, None): {'lam_v': 7.3e5, 'lam_w': 1.5e6},
    (5, None): {'lam_v': 1.1e6, 'lam_w': 2.4e6},
    (3, 10):   {'lam_v': 1.2e7, 'lam_w': 2.6e8},
    (3, 20):   {'lam_v': 1.2e7, 'lam_w': 3.7e7},
    (3, 30):   {'lam_v': 5.9e6, 'lam_w': 7.2e6},
}

# --- MART (smart2) ---
# Global dataset averages for the SMART2 structural prior
SMART2_PRIOR_PARAMS = {
    'fitter': 'mpfit',
    'psi': 0.2,
    'prior_weight': 1.0,
    'maxouter': 5,
    'maxinner': 20,
    'frac1': 0.8555,
    'frac2': 0.0521,
    'frac_bg': 0.0924,
    'cent1': -1.13*(195.11794/299792.458)+195.11803,
    'wid1': 42.74*(195.11794/299792.458), # Using the pure microscopic width!
    'cent2': 195.17803,
    'wid2': 42.74*(195.11794/299792.458),
    'bg_shape_norm': [0.04762] * 21
}

# --- U-Net (nn_solver) ---
# Put the exact folder names of your trained U-Net models for each configuration
UNET_MODEL_PATHS = {
    (2, None): '2026_05_11__17_27_34_NF_64_BS_4_LR_0.0002_EP_400_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_100_None_K_2_eis_v5',
    (3, None): '2026_05_11__17_26_39_NF_64_BS_4_LR_0.0002_EP_400_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_100_None_K_3_eis_v5',
    (4, None): '2026_05_11__17_27_59_NF_64_BS_4_LR_0.0002_EP_400_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_100_None_K_4_eis_v5',
    (5, None): '2026_05_11__17_29_26_NF_64_BS_4_LR_0.0002_EP_400_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_100_None_K_5_eis_v5',
    (3, 10):   '2026_05_13__05_02_23_NF_64_BS_4_LR_0.0002_EP_100_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_10_poisson_K_3_eis_v5',
    (3, 20):   '2026_05_13__05_05_48_NF_64_BS_4_LR_0.0002_EP_100_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_20_poisson_K_3_eis_v5',
    (3, 30):   '2026_05_13__05_06_25_NF_64_BS_4_LR_0.0002_EP_100_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_30_poisson_K_3_eis_v5',
}

# ==============================================================================
# 3. AUTOMATED RUNNER SCRIPT
# ==============================================================================
def run_all_configs():
    print(f"Loading Dataset: {DATA_FILE}...")
    data = np.load(os.path.join(PATH_DATA, DATA_FILE), allow_pickle=True).item()
    param4dar = data['param3d']
    meas_key = 'meas_damped' if 'meas_damped' in data else 'meas'
    
    savepath = str(config.results_root / 'recons')
    now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    master_savedir = os.path.join(savepath, f'{now}_final_runner_{METHOD.upper()}')
    os.makedirs(master_savedir, exist_ok=True)
    print(f"\nAll plots, logs, and Pickles will be securely saved to:\n  -> {master_savedir}/\n")
    
    # Accumulate the results here so we can print the LaTeX table at the end
    results_summary = {}
    
    total_configs = len(CONFIGS)
    global_start_time = time.time()

    for idx, (K, dbsnr) in enumerate(CONFIGS):
        print("\n" + "="*70)
        snr_str = f"{dbsnr} dB" if dbsnr is not None else "Noiseless"
        print(f" EVALUATING CONFIGURATION {idx+1}/{total_configs}: K={K}, SNR={snr_str}")
        print("="*70)
        
        # 1. Setup Imager
        spectral_orders = [0, -1, 1, -2, 2][:K]
        meas4dar = data[meas_key][:, :K]
        
        noise_model = 'poisson' if dbsnr is not None else None
        avg_count = dbsnr**2 if dbsnr is not None else None
        
        Imgr = Imager(
            pixelated=True, 
            spectral_orders=spectral_orders, 
            dispersion_scale=0.022275,
            mid_wavelength=195.119,
            dbsnr=dbsnr,
            noise_model=noise_model,
            avg_count=avg_count
        )
        
        # 2. Setup Solver specific arguments
        kwargs = {}
        if METHOD == '1dmap':
            solver = scipy_solver_parallel2
            kwargs = {
                'OPTIMIZER': 'L-BFGS-B',
                'DATA_FIDELITY': 'L2',
                'lam_i': 1e-4,
                'maxiter': 10000
            }
            kwargs.update(OPTIMAL_PARAMS_1DMAP[(K, dbsnr)])
            
        elif METHOD == 'mart':
            solver = smart2_twostage
            kwargs = SMART2_PRIOR_PARAMS.copy()
            
        elif METHOD == 'unet':
            solver = nn_solver
            kwargs = {
                'model_path': UNET_MODEL_PATHS[(K, dbsnr)]
            }
        else:
            raise ValueError("Invalid METHOD selected. Choose '1dmap', 'mart', or 'unet'.")

        # 3. Initialize and Run Reconstructor
        Rec = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            meas4dar=meas4dar,
            pix=False, # True source is physical
            solver=solver,
            intenscaling=False,
            **kwargs
        )
        
        print(f"Running {METHOD.upper()} over {len(param4dar)} patches...")
        Rec.solve(num_realizations=1)
        
        # 4. Extract Metrics
        # (Note: Rec.rmse_phy converts the width to km/s automatically during eval)
        rmse_phy = Rec.rmse_phy.mean(axis=(0,1))
        bias_phy = Rec.bias_phy.mean(axis=(0,1))
        
        results_summary[(K, dbsnr)] = {
            'rmse': rmse_phy,
            'bias': bias_phy,
            'time': Rec.times.mean()
        }
        
        print(f"  Int RMSE: {rmse_phy[0]:.2f} | Bias: {bias_phy[0]:.2f}")
        print(f"  Vel RMSE: {rmse_phy[1]:.3f} | Bias: {bias_phy[1]:.3f}")
        print(f"  Wid RMSE: {rmse_phy[2]:.3f} | Bias: {bias_phy[2]:.3f}")
        
        elapsed_time = time.time() - global_start_time
        avg_time_per_config = elapsed_time / (idx + 1)
        eta_seconds = avg_time_per_config * (total_configs - (idx + 1))
        print(f"\n  -> [Elapsed: {datetime.timedelta(seconds=int(elapsed_time))} | ETA: {datetime.timedelta(seconds=int(eta_seconds))}]")

        # ==========================================================================
        # 5. SAVING ROUTINE (Extract, Plot, and Dump Pickles)
        # ==========================================================================
        config_name = f'K_{K}_{noise_model}_dbsnr_{dbsnr}'
        savedir = os.path.join(master_savedir, config_name)
        os.makedirs(savedir, exist_ok=True)
        
        recon_summary = [
            '############## Recon Parameters ############## \n',
            'Solver: {} \n'.format(Rec.solver.__name__),
            'Num Detectors: {} \n'.format(K),
            'Noise Model / dbsnr: {} / {} \n'.format(noise_model, dbsnr),
            'Num Realizations: {} \n'.format(Rec.num_realizations),
            'Solver Params: {} \n'.format(Rec.solver_params),
            'Recon Time Avg: {:.2f} s \n'.format(Rec.times.mean()),
            'RMSE_phy Avg (per Img): {} \n'.format(Rec.rmse_phy.mean(axis=1)),
            'RMSE_phy Avg: {} \n'.format(Rec.rmse_phy.mean(axis=(0,1))),
            'MAE_phy Avg: {} \n'.format(Rec.mae_phy.mean(axis=(0,1))),
            'Bias_phy Avg (per Img): {} \n'.format(Rec.bias_phy.mean(axis=1)),
            'Bias_phy Avg: {} \n'.format(Rec.bias_phy.mean(axis=(0,1))),
            'RMSE_pix Avg (per Img): {} \n'.format(Rec.rmse_pix.mean(axis=1)),
            'RMSE_pix Avg: {} \n'.format(Rec.rmse_pix.mean(axis=(0,1)))
        ]

        recons = Rec.recons
        rec_array_pix = []
        truth_array_pix = []
        
        print(f"  Saving {len(recons)} plots and exporting evaluation metrics...")
        for i in range(len(recons)):
            fig, ax = recons[i].plot(compare=True, title=f'{Rec.solver.__name__}')
            fig.savefig(os.path.join(savedir, f'recon_{i}.png'))
            plt.close(fig) # Close figure immediately to prevent memory leaks!
            rec_array_pix.append(recons[i].recon)
            truth_array_pix.append(Rec.sources[i].param3d[None])
            
        rec_array_pix = np.array(rec_array_pix)
        truth_array_pix = np.array(truth_array_pix)
        rec_array_phy = Rec.imager.frompix(rec_array_pix, width_unit='km/s', array=True)
        truth_array_phy = Rec.imager.frompix(truth_array_pix, width_unit='km/s', array=True)
        diff_pix = rec_array_pix - truth_array_pix
        diff_phy = rec_array_phy - truth_array_phy

        def hist_plotter(diff, unit='phy'):
            unitstr = 'km/s' if unit=='phy' else 'pixels'
            fig, ax = plt.subplots(1,3, figsize=(13.8,4.8))
            ax[0].hist(diff[:,:,0].flatten(), bins=20, edgecolor='Black', color='sandybrown')
            ax[0].set_title(('Intensity RMS Error = {:.4f} \n '.format(np.sqrt(np.mean(diff[:,:,0]**2))) +
            r'Intensity Bias = {:.4f}'.format(np.mean(diff[:,:,0]))))
            ax[0].set_xlabel('Error')
            ax[0].set_ylabel('Number of Occurances')
            ax[1].hist(diff[:,:,1].flatten(), bins=20, edgecolor='Black', color='sandybrown')
            ax[1].set_title(('Doppler Velocity RMS Error = {:.3f} {} \n'.format(np.sqrt(np.mean(diff[:,:,1]**2)),unitstr)+
            'Doppler Velocity Bias = {:.3f} {}'.format(np.mean(diff[:,:,1]),unitstr)))
            ax[1].set_xlabel(f'Error [{unitstr}]')
            ax[1].set_ylabel('Number of Occurances')
            ax[2].hist(diff[:,:,2].flatten(), bins=20, edgecolor='Black', color='sandybrown')
            ax[2].set_title(('Line Width RMS Error = {:.3f} {} \n'.format(np.sqrt(np.mean(diff[:,:,2]**2)), unitstr)+
            'Line Width Bias = {:.3f} {}'.format(np.mean(diff[:,:,2]), unitstr)))
            ax[2].set_xlabel(f'Error [{unitstr}]')
            ax[2].set_ylabel('Number of Occurances')
            plt.tight_layout()
            return fig, ax

        fig_phy, ax_phy = hist_plotter(diff_phy, 'phy')
        fig_phy.savefig(os.path.join(savedir, 'error_hist_phy.png'))
        plt.close(fig_phy)
        
        fig_pix, ax_pix = hist_plotter(diff_pix, 'pix')
        fig_pix.savefig(os.path.join(savedir, 'error_hist_pix.png'))
        plt.close(fig_pix)

        with open(os.path.join(savedir, 'summary.txt'), 'w') as file:
            for line in recon_summary:
                file.write(line)

        with open(os.path.join(savedir, 'Rec.pickle'), 'wb') as file:
            pickle.dump(Rec, file)

    # ==============================================================================
    # 4. PRINT LATEX TABLE ROWS
    # ==============================================================================
    print("\n\n" + "="*80)
    print(f" FINAL LATEX ROWS FOR {METHOD.upper()} ")
    print("="*80)
    
    print("\n--- TABLE 1 (Varying K at Native SNR) ---")
    for K in [2, 3, 4, 5]:
        res = results_summary.get((K, None))
        if res:
            rmse, bias = res['rmse'], res['bias']
            print(f"\\multirow{{3}}{{*}}{{{K}}} & {METHOD.upper()} & {rmse[0]:.1f} & {bias[0]:.1f} & {rmse[1]:.3f} & {bias[1]:.3f} & {rmse[2]:.3f} & {bias[2]:.3f} \\\\")
            
    print("\n--- TABLE 2 (Varying SNR at K=3) ---")
    for snr in [10, 20, 30, None]:
        res = results_summary.get((3, snr))
        if res:
            rmse, bias = res['rmse'], res['bias']
            snr_label = "\\infty" if snr is None else snr
            print(f"\\multirow{{3}}{{*}}{{{snr_label}}} & {METHOD.upper()} & {rmse[0]:.1f} & {bias[0]:.1f} & {rmse[1]:.3f} & {bias[1]:.3f} & {rmse[2]:.3f} & {bias[2]:.3f} \\\\")

    # ==============================================================================
    # 5. SAVE OVERALL SUMMARY AND LATEX
    # ==============================================================================
    summary_path = os.path.join(master_savedir, 'results_summary.pickle')
    with open(summary_path, 'wb') as f:
        pickle.dump(results_summary, f)
    print(f"\nSaved overall results summary to: {summary_path}")

    latex_path = os.path.join(master_savedir, 'latex_tables.txt')
    with open(latex_path, 'w') as f:
        f.write("--- TABLE 1 (Varying K at Native SNR) ---\n")
        for K in [2, 3, 4, 5]:
            res = results_summary.get((K, None))
            if res:
                rmse, bias = res['rmse'], res['bias']
                f.write(f"\\multirow{{3}}{{*}}{{{K}}} & {METHOD.upper()} & {rmse[0]:.1f} & {bias[0]:.1f} & {rmse[1]:.3f} & {bias[1]:.3f} & {rmse[2]:.3f} & {bias[2]:.3f} \\\\\n")
        f.write("\n--- TABLE 2 (Varying SNR at K=3) ---\n")
        for snr in [10, 20, 30, None]:
            res = results_summary.get((3, snr))
            if res:
                rmse, bias = res['rmse'], res['bias']
                snr_label = "\\infty" if snr is None else snr
                f.write(f"\\multirow{{3}}{{*}}{{{snr_label}}} & {METHOD.upper()} & {rmse[0]:.1f} & {bias[0]:.1f} & {rmse[1]:.3f} & {bias[1]:.3f} & {rmse[2]:.3f} & {bias[2]:.3f} \\\\\n")
    print(f"Saved LaTeX tables to: {latex_path}")

    txt_path = os.path.join(master_savedir, 'results_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f" FINAL EVALUATION SUMMARY - METHOD: {METHOD.upper()}\n")
        f.write("="*60 + "\n\n")
        for K, dbsnr in CONFIGS:
            res = results_summary.get((K, dbsnr))
            if res:
                snr_str = f"{dbsnr} dB" if dbsnr is not None else "Noiseless"
                f.write(f"Configuration: K={K}, SNR={snr_str}\n")
                f.write("-" * 40 + "\n")
                rmse, bias, time_val = res['rmse'], res['bias'], res['time']
                f.write(f"  Int RMSE: {rmse[0]:8.2f} | Bias: {bias[0]:8.2f}\n")
                f.write(f"  Vel RMSE: {rmse[1]:8.3f} | Bias: {bias[1]:8.3f} km/s\n")
                f.write(f"  Wid RMSE: {rmse[2]:8.3f} | Bias: {bias[2]:8.3f} km/s\n")
                f.write(f"  Avg Time: {time_val:.2f} s\n\n")
    print(f"Saved human-readable summary to: {txt_path}")

    total_time = time.time() - global_start_time
    print(f"\nAll configurations completed successfully in {datetime.timedelta(seconds=int(total_time))}!")

if __name__ == '__main__':
    run_all_configs()
