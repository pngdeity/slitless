import os
import glob
import numpy as np
from slitless.forward import forward_op_tomo_3d
from slitless.config import config

def generate_consolidated_set(num_patches=50, split='train', output_filename=None):
    if output_filename is None:
        output_filename = f'eis_{split}_{num_patches}_dsetv5.npy'
        
    split_dir = str(config.data_root / f'eis_data/datasets/dset_v5/data/{split}')
    out_dir = str(config.data_root / 'datasets/baseline')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    
    print(f"Scanning {split_dir}...")
    files = glob.glob(os.path.join(split_dir, '*.npy'))
    files.sort()  # Ensure reproducible ordering across different operating systems
    
    # Set a seed to ensure reproducibility if you run this multiple times
    np.random.seed(42)
    selected_files = np.random.choice(files, num_patches, replace=False)
    
    param3d_l, meas_l, meas2_l, cube_l = [], [], [], []
    
    print(f"Stacking {num_patches} randomly selected patches using small_train_generator logic...")
    for f in selected_files:
        data_i = np.load(f, allow_pickle=True).item()
        meas_l.append(np.stack([data_i['meas_0'], data_i['meas_-1'], data_i['meas_1'], data_i['meas_-2'], data_i['meas_2']]))
        param3d_l.append(np.stack([data_i['int'], data_i['vel'], data_i['width']]))
        cube_l.append(data_i['datacube'])
        meas2_l.append(forward_op_tomo_3d(data_i['datacube'].transpose(2,0,1), orders=[0,-1,1,-2,2]))
        
    out_data = {
        'meas': np.array(meas_l),
        'meas_damped': np.array(meas2_l),
        'param3d': np.array(param3d_l),
        'datacube': np.array(cube_l)
    }
    
    np.save(out_path, out_data)
    print(f"\nSuccessfully saved to {out_path}")
    print(f"  param3d shape:     {out_data['param3d'].shape}")
    print(f"  meas_damped shape: {out_data['meas_damped'].shape}")
    print(f"  datacube shape:    {out_data['datacube'].shape}")

if __name__ == '__main__':
    generate_consolidated_set(num_patches=1000, split='train')
    generate_consolidated_set(num_patches=100, split='test')
