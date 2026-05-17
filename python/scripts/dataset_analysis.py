import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from slitless.config import config

# --- CONFIG ---
DATA_DIR = str(config.data_root / 'eis_data/datasets/dset_v4/data/train')
SAVE_DIR = str(config.data_root / 'eis_data/datasets/dset_v4/analysis')
os.makedirs(SAVE_DIR, exist_ok=True)

# SAMPLE_RATE: Fraction of files to load (0.05 = 5%). 
# 100% is too big for RAM (140 million pixels).
SAMPLE_RATE = 0.5 

def get_percentiles(arr):
    return np.nanpercentile(arr, [0, 1, 5, 50, 95, 99, 100])

print(f"Scanning {DATA_DIR}...")
all_files = glob.glob(os.path.join(DATA_DIR, "*.npy"))
print(f"Found {len(all_files)} files. Sampling {int(len(all_files)*SAMPLE_RATE)}...")

# Shuffle and slice to get random sample
np.random.shuffle(all_files)
sample_files = all_files[:int(len(all_files)*SAMPLE_RATE)]

# Storage
raw_data = {'int': [], 'vel': [], 'wid': []}
clean_data = {'int': [], 'vel': [], 'wid': []}
masked_scan_cnt = 0

for f in tqdm(sample_files):
    d = np.load(f, allow_pickle=True).item()
    
    # Flatten arrays
    i_flat = d['int'].flatten()
    v_flat = d['vel'].flatten()
    w_flat = d['width'].flatten()
    
    # 1. RAW DATA (Just remove NaNs/Infs)
    valid_raw = np.isfinite(i_flat) & np.isfinite(v_flat)
    raw_data['int'].append(i_flat[valid_raw])
    raw_data['vel'].append(v_flat[valid_raw])
    raw_data['wid'].append(w_flat[valid_raw])
    
    # 2. FILTERED DATA (The "Physics" Mask)
    # Intensity > 5, |Velocity| < 300, 0.04 < Width < 0.2
    mask = (
        (i_flat > 100) & 
        (i_flat < 6000) & 
        (np.abs(v_flat) <= 68.5) &
        (w_flat >= 0.022)
    )
    
    if mask.sum() > 0:
        clean_data['int'].append(i_flat[mask])
        clean_data['vel'].append(v_flat[mask])
        clean_data['wid'].append(w_flat[mask])
    if mask.sum() < len(i_flat):
    # if mask.sum() < 2000:
        masked_scan_cnt += 1

# Concatenate
print("\nConcatenating arrays...")
R_int = np.concatenate(raw_data['int'])
R_vel = np.concatenate(raw_data['vel'])
R_wid = np.concatenate(raw_data['wid'])

C_int = np.concatenate(clean_data['int'])
C_vel = np.concatenate(clean_data['vel'])
C_wid = np.concatenate(clean_data['wid'])

print(f"Total Pixels Analyzed: {len(R_int)}")
print(f"Clean Physics Pixels:  {len(C_int)} ({len(C_int)/len(R_int)*100:.1f}%)")
print(f"Discarded Scan Rate: {masked_scan_cnt}/{len(sample_files)} ({masked_scan_cnt/len(sample_files)*100:.1f}%)")


# --- STATS REPORT ---
def print_stats(name, raw, clean):
    r_p = get_percentiles(raw)
    c_p = get_percentiles(clean)
    print(f"\n--- {name} STATISTICS ---")
    print(f"{'Metric':<10} | {'Raw (All Pixels)':<20} | {'Clean (Physics Mask)':<20}")
    print("-" * 60)
    print(f"{'Min':<10} | {r_p[0]:<20.4f} | {c_p[0]:<20.4f}")
    print(f"{'1%':<10} | {r_p[1]:<20.4f} | {c_p[1]:<20.4f}")
    print(f"{'Median':<10} | {r_p[3]:<20.4f} | {c_p[3]:<20.4f}")
    print(f"{'99%':<10} | {r_p[5]:<20.4f} | {c_p[5]:<20.4f}")
    print(f"{'Max':<10} | {r_p[6]:<20.4f} | {c_p[6]:<20.4f}")
    print(f"{'Mean':<10} | {np.nanmean(raw):<20.4f} | {np.nanmean(clean):<20.4f}")
    print(f"{'Std':<10} | {np.nanstd(raw):<20.4f} | {np.nanstd(clean):<20.4f}")

print_stats("INTENSITY", R_int, C_int)
print_stats("VELOCITY", R_vel, C_vel)
print_stats("WIDTH", R_wid, C_wid)

# --- VISUALIZATION ---
print("\nGenerating Plots...")

# 1. Velocity Distribution (The most important one)
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(C_vel, bins=100, range=(-20, 20), color='blue', alpha=0.7, label='Clean')
# Overlay Gaussian
mu, std = np.mean(C_vel), np.std(C_vel)
ax.set_title('Clean Velocity Distribution')
ax.set_xlabel('km/s')
ax.legend()
plt.savefig(SAVE_DIR + 'dist_velocity.png')
plt.close()

# 2. Intensity Distribution (Log Scale)
plt.figure(figsize=(8, 5))
plt.hist(np.log10(C_int), bins=100, color='orange', alpha=0.7, label='Clean')
plt.title('Intensity Distribution (Log10)')
plt.xlabel('Log10(Intensity)')
plt.ylabel('Count')
plt.legend()
plt.savefig(SAVE_DIR + 'dist_intensity_log.png')
plt.close()

# 2. Intensity Distribution
plt.figure(figsize=(8, 5))
plt.hist(C_int, bins=100, color='orange', alpha=0.7, label='Clean')
plt.title('Intensity Distribution')
plt.xlabel('Intensity')
plt.ylabel('Count')
plt.legend()
plt.savefig(SAVE_DIR + 'dist_intensity.png')
plt.close()

# 3. Width Distribution
plt.figure(figsize=(8, 5))
plt.hist(C_wid, bins=100, color='green', alpha=0.7, label='Clean')
plt.axvline(x=0.022, color='red', linestyle='--', label='Inst. Width (~0.022)')
plt.title('Spectral Width Distribution')
plt.xlabel('Angstroms')
plt.legend()
plt.savefig(SAVE_DIR + 'dist_width.png')
plt.close()

# 4. Joint Plot: Velocity vs Intensity (2D Histogram)
# This proves if low intensity = high noise
plt.figure(figsize=(10, 6))
# Use clean intensity but slightly relaxed velocity to see the noise fan
mask_joint = (R_int > 1.0) & (np.abs(R_vel) < 500)
plt.hist2d(np.log10(R_int[mask_joint]), R_vel[mask_joint], bins=(100, 100), cmap='inferno', norm=LogNorm())
plt.colorbar(label='Pixel Count')
plt.title('Velocity Noise vs. Signal Strength')
plt.xlabel('Log10(Intensity)')
plt.ylabel('Velocity (km/s)')
plt.ylim(-400, 400)
plt.savefig(SAVE_DIR + 'joint_vel_int.png')
plt.close()

print(f"Analysis Complete. Check plots in {SAVE_DIR}")