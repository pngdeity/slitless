import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
from slitless.config import config

def plot_theoretical_margin_of_error():
    # Physical Constants
    C = 299792.458
    REST_WAVE = 195.119
    Z = 1.96  # 95% Confidence Level
    
    # Global Standard Deviations from norm_stats.npy
    sigma_vel = 9.061165  # km/s
    sigma_wid_A = 0.001396 # Angstroms
    sigma_wid = sigma_wid_A * (C / REST_WAVE) # km/s
    
    # Sample sizes to evaluate
    n_samples = np.arange(1, 101, 1)
    
    # Calculate Margins of Error
    moe_vel = Z * (sigma_vel / np.sqrt(n_samples))
    moe_wid = Z * (sigma_wid / np.sqrt(n_samples))
    
    plt.figure(figsize=(10, 5))
    plt.plot(n_samples, moe_vel, label='Velocity MOE (km/s)', color='red', lw=2)
    plt.plot(n_samples, moe_wid, label='Line Width MOE (km/s)', color='blue', lw=2)
    
    plt.axhline(0.5, color='gray', linestyle='--', label='0.5 km/s Tolerance')
    plt.axhline(0.1, color='lightgray', linestyle='--', label='0.1 km/s Tolerance')
    
    plt.title('Theoretical 95% Confidence Margin of Error vs Independent Samples')
    plt.xlabel('Number of Independent Samples (n)')
    plt.ylabel('Margin of Error (km/s)')
    plt.ylim(0, 2.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('theoretical_sample_size.png', dpi=200)
    print("Saved 'theoretical_sample_size.png'")

def plot_empirical_convergence():
    train_dir = str(config.data_root / 'eis_data/datasets/dset_v5/data/train')
    files = glob.glob(os.path.join(train_dir, '*.npy'))
    
    # Load a pool of up to 1000 patches to test convergence
    np.random.shuffle(files)
    files = files[:1000]
    
    C = 299792.458
    REST_WAVE = 195.119
    
    vel_means = []
    wid_means = []
    
    print("Loading patches to simulate test-set convergence...")
    for f in tqdm(files):
        d = np.load(f, allow_pickle=True).item()
        vel_means.append(np.mean(d['vel']))
        wid_means.append(np.mean(d['width']) * (C / REST_WAVE)) # Convert to km/s
        
    n_samples_list = np.arange(1, 101, 1)
    empirical_std_vel = []
    empirical_std_wid = []
    
    # Bootstrap to find the empirical standard deviation of the mean
    for n in n_samples_list:
        v_boot = [np.mean(np.random.choice(vel_means, n, replace=True)) for _ in range(200)]
        w_boot = [np.mean(np.random.choice(wid_means, n, replace=True)) for _ in range(200)]
        empirical_std_vel.append(np.std(v_boot))
        empirical_std_wid.append(np.std(w_boot))
        
    print("\n--- Bootstrapped Standard Deviations ---")
    print(f"n_samples_list = {list(n_samples_list)}\n")
    print(f"empirical_std_vel = {[float(f'{v:.3g}') for v in empirical_std_vel]}\n")
    print(f"empirical_std_wid = {[float(f'{v:.3g}') for v in empirical_std_wid]}\n")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(n_samples_list, empirical_std_vel, color='red', lw=2)
    axes[0].axhline(0.2, color='gray', linestyle='--', label='0.2 km/s Tolerance')
    axes[0].set_title('Empirical Standard Deviation of Velocity Mean vs Number of Patches')
    axes[0].set_ylabel('Std Dev of Mean (km/s)')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(n_samples_list, empirical_std_wid, color='blue', lw=2)
    axes[1].axhline(0.1, color='gray', linestyle='--', label='0.1 km/s Tolerance')
    axes[1].set_title('Empirical Standard Deviation of Line Width Mean vs Number of Patches')
    axes[1].set_xlabel('Number of 64x64 Patches in Test Set')
    axes[1].set_ylabel('Std Dev of Mean (km/s)')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('empirical_patch_convergence.png', dpi=200)
    print("Saved 'empirical_patch_convergence.png'")

if __name__ == '__main__':
    plot_theoretical_margin_of_error()
    plot_empirical_convergence()