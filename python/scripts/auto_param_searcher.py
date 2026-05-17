import os
import numpy as np
import optuna
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis
from slitless.forward import Imager
from slitless.recon import scipy_solver_parallel, scipy_solver_parallel2, Reconstructor_Multi
from slitless.config import config

# 1. Define Data Setup
path_data = str(config.data_root / 'datasets/baseline')
data_file = 'eis_train_5_dsetv5.npy' 

print(f"Loading {data_file}...")
data = np.load(path_data + data_file, allow_pickle=True).item()

param4dar = data['param3d']
# Use all 5 images with real measurements
meas_key = 'meas_damped' if 'meas_damped' in data else 'meas'

source_pix = False
intenscaling = False  # Keep False to bypass optimizer lock

M = param4dar.shape[-1]

# 2. Define Objective Function for Optuna
def create_objective(current_num_detectors):
    # Define specific orders for the current detector count
    spectral_orders = [0, -1, 1, -2, 2][:current_num_detectors]
    
    # Slice measurements accordingly
    current_meas4dar = data[meas_key][:, :current_num_detectors]

    def objective(trial):
        # Broaden search ranges to safely accommodate high and low noise levels
        lam_v = trial.suggest_float('lam_v', 1e1, 1e10, log=True)
        lam_w = trial.suggest_float('lam_w', 1e1, 1e10, log=True)
        
        # Initialize Imager cleanly for the noiseless setup
        Imgr = Imager(
            pixelated=True, 
            spectral_orders=spectral_orders, 
            dispersion_scale=0.022275,
            mid_wavelength=195.119,
            dbsnr=None,          # Noiseless
            noise_model=None,    # Noiseless
            avg_count=None       # Noiseless
        )

        # SCIPY
        Rec = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            meas4dar=current_meas4dar, # Pass correctly sized tomographic measurements
            pix=source_pix,
            solver=scipy_solver_parallel2,
            intenscaling=intenscaling,
            DATA_FIDELITY='L2',
            OPTIMIZER='L-BFGS-B',
            maxiter=10000,      # Slightly lowered to accelerate parameter search
            lam_i=1e-4,
            lam_v=lam_v,
            lam_w=lam_w,
            frac1=0.8620,
            frac2=0.0521,
            frac_bg=0.0860,
            cent1=195.11723,
            wid1=0.02981,
            cent2=195.17723,
            wid2=0.02981,
            bg_shape_norm=[0.04762] * 21
        )

        # Run the solver
        Rec.solve(num_realizations=1)
        
        # Extract the physical RMSE for Velocity and Line Width
        rmse_vel = Rec.rmse_phy.mean(axis=(0, 1))[1]
        rmse_wid = Rec.rmse_phy.mean(axis=(0, 1))[2]
        
        # Target function minimizes the combined errors
        combined_rmse = rmse_vel + rmse_wid
        
        print(f"Trial {trial.number:02d} | lam_v: {lam_v:.2e}, lam_w: {lam_w:.2e} --> "
              f"Vel RMSE: {rmse_vel:.4f} km/s | Wid RMSE: {rmse_wid:.4f} km/s | Combined: {combined_rmse:.4f}")
        
        return combined_rmse
    return objective

if __name__ == '__main__':
    import json
    print("Starting Automated Parameter Search for Multiple Detector Counts (Noiseless)...")
    
    # Suppress Optuna's default verbose logging to keep the console clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    detectors_list = [2, 3, 4, 5]
    results_summary = {}
    
    for current_num_detectors in detectors_list:
        print("\n" + "="*50)
        print(f" SEARCHING OPTIMAL PARAMS FOR num_detectors = {current_num_detectors} (Noiseless)")
        print("="*50)
        
        study = optuna.create_study(direction='minimize')
        
        # Evaluate 30 intelligent trials per detector setup
        study.optimize(create_objective(current_num_detectors), n_trials=30)
        
        print(f"\nBest parameters found for num_detectors={current_num_detectors}:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value:.2e}")
        print(f"Resulting Best Combined Vel+Wid RMSE: {study.best_value:.4f}")
        
        results_summary[f'num_detectors_{current_num_detectors}'] = {
            'lam_v': study.best_params['lam_v'],
            'lam_w': study.best_params['lam_w'],
            'best_rmse': study.best_value
        }
        
        print(f"Saving Optuna Analysis Plots for num_detectors={current_num_detectors}...")
        
        # Optimization History Plot
        ax1 = vis.plot_optimization_history(study)
        ax1.figure.tight_layout()
        ax1.figure.savefig(f'optuna_history_K_{current_num_detectors}_noiseless.png', dpi=150)
        
        # Contour Plot (Landscape)
        ax2 = vis.plot_contour(study, params=['lam_v', 'lam_w'])
        ax2.figure.tight_layout()
        ax2.figure.savefig(f'optuna_contour_K_{current_num_detectors}_noiseless.png', dpi=150)
        
        plt.close('all') # Clear plots memory to prevent overlap between SNR loops

    print("\n" + "="*50)
    print(" FINAL SUMMARY OF OPTIMAL PARAMETERS ")
    print("="*50)
    for snr_key, res in results_summary.items():
        print(f"{snr_key}: lam_v = {res['lam_v']:.2e}, lam_w = {res['lam_w']:.2e} (RMSE = {res['best_rmse']:.4f})")
        
    with open('optimal_scipy2_params_k.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    print("\nSaved summary to 'optimal_scipy2_params_k.json'")
