import os
import glob
import numpy as np
from tqdm import tqdm
from slitless.forward import forward_op_tomo_3d
from slitless.config import config

# --- CONFIGURATION ---
OUT_DIR = str(config.data_root / 'eis_data/datasets/dset_v5/data')
STATS_OUT_PATH = str(config.data_root / 'eis_data/datasets/dset_v5/norm_stats.npy')

DISP_SCALE_A = 0.022275  # Angstroms per pixel
WAVELENGTH = 195.119     # Angstroms
LAMDIM = 21

def main():
    splits = ['train', 'val', 'test']
    
    for split in splits:
        out_split_dir = os.path.join(OUT_DIR, split)
        os.makedirs(out_split_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(out_split_dir, '*.npy'))
        print(f"\nProcessing {split} split ({len(files)} files)...")
        
        for file_path in tqdm(files):
            filename = os.path.basename(file_path)
            out_file_path = os.path.join(out_split_dir, filename)
            
            # Load existing dset_v5 data
            data = np.load(file_path, allow_pickle=True).item()
            cube = data['datacube']
            
            # Generate tomographic measurements from the datacube
            meas_tomo = forward_op_tomo_3d(cube.transpose(2,0,1), orders=[0,-1,1,-2,2])
            
            # Overwrite the original measurements with the tomographic ones
            data['meas_0'] = meas_tomo[0]
            data['meas_-1'] = meas_tomo[1]
            data['meas_1'] = meas_tomo[2]
            data['meas_-2'] = meas_tomo[3]
            data['meas_2'] = meas_tomo[4]
            
            # Save updated dictionary to dset_v5
            np.save(out_file_path, data)

    print("\n--- Generating new norm_stats.npy for dset_v5 ---")
    train_files = glob.glob(os.path.join(OUT_DIR, 'train', '*.npy'))
    
    int_list, vel_list, wid_list, meas_list = [], [], [], []
    
    for file_path in tqdm(train_files):
        d = np.load(file_path, allow_pickle=True).item()
        int_list.append(d['int'])
        vel_list.append(d['vel'])
        wid_list.append(d['width'])
        meas_list.append(d['meas_0']) # Sample meas for meas_mean
        
    int_arr = np.stack(int_list)
    vel_arr = np.stack(vel_list)
    wid_arr = np.stack(wid_list)
    meas_arr = np.stack(meas_list)
    
    stats = {
        'int_max': np.max(int_arr),
        'int_mean': np.mean(int_arr),
        'meas_mean': np.mean(meas_arr),
        'vel_mean': np.mean(vel_arr),
        'vel_std': np.std(vel_arr),
        'width_mean': np.mean(wid_arr),
        'width_std': np.std(wid_arr)
    }
    
    np.save(STATS_OUT_PATH, stats)
    print(f"Stats saved to {STATS_OUT_PATH}")
    print(stats)

if __name__ == '__main__':
    main()
