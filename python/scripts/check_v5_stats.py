import numpy as np
import glob
import os
from tqdm import tqdm
from slitless.config import config

stats_path = str(config.data_root / 'eis_data/datasets/dset_v5/norm_stats.npy')
train_dir = str(config.data_root / 'eis_data/datasets/dset_v5/data/train')

print(f"Loading stats from: {stats_path}")
saved_stats = np.load(stats_path, allow_pickle=True).item()

print("\n--- SAVED STATS ---")
for k, v in saved_stats.items():
    print(f"  {k}: {v:.6f}")

print(f"\nVerifying against ALL training data from {train_dir}...")
files = glob.glob(os.path.join(train_dir, '*.npy'))

int_list, vel_list, wid_list = [], [], []
for f in tqdm(files):
    d = np.load(f, allow_pickle=True).item()
    int_list.append(d['int'])
    vel_list.append(d['vel'])
    wid_list.append(d['width'])

int_arr = np.stack(int_list)
vel_arr = np.stack(vel_list)
wid_arr = np.stack(wid_list)

print("\n--- CALCULATED STATS (Entire Training Set) ---")
print(f"  int_max: {np.max(int_arr):.6f} \t (Saved: {saved_stats['int_max']:.6f})")
print(f"  int_mean: {np.mean(int_arr):.6f} \t (Saved: {saved_stats['int_mean']:.6f})")
print(f"  vel_mean: {np.mean(vel_arr):.6f} \t (Saved: {saved_stats['vel_mean']:.6f})")
print(f"  vel_std: {np.std(vel_arr):.6f} \t (Saved: {saved_stats['vel_std']:.6f})")
print(f"  width_mean: {np.mean(wid_arr):.6f} \t (Saved: {saved_stats['width_mean']:.6f})")
print(f"  width_std: {np.std(wid_arr):.6f} \t (Saved: {saved_stats['width_std']:.6f})")
