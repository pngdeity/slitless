import numpy as np
import glob, os
from torch.utils.data import Dataset, DataLoader
from slitless.data_loader import BasicDataset
from slitless.config import config
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
base_dir = str(config.data_root / 'eis_data/datasets/dset_v4/data')
train_dir = str(config.data_root / 'eis_data/datasets/dset_v4/data/train')

def stat_calculator():
    trainset = BasicDataset(base_dir)
    trainloader = DataLoader(trainset, batch_size=len(trainset), num_workers=8)
    data = next(iter(trainloader))
    params = data[1]
    meas = data[0]

    int_mean, vel_mean, width_mean = params.mean(dim=(0,2,3)).numpy()
    _, vel_std, width_std = params.std(dim=(0,2,3)).numpy()
    int_max = params.amax(dim=(0,2,3))[0].numpy()
    meas_mean = meas.mean().numpy()

    # 3. Compute Stats on CLEAN data
    stats = {
        'int_max': int_max,
        'int_mean': int_mean,
        'meas_mean': meas_mean,
        'vel_mean': vel_mean,
        'vel_std': vel_std,
        'width_mean': width_mean,
        'width_std': width_std
    }

    print("\n--- ROBUST STATS ---")
    print(stats)
    np.save(str(config.data_root / 'eis_data/datasets/dset_v4/norm_stats.npy'), stats)

def train_test_generator(base_dir=base_dir):
    # Destination folders (Must exist, or script will create them)
    train_dir = os.path.join(base_dir, 'train')
    val_dir   = os.path.join(base_dir, 'val')
    test_dir  = os.path.join(base_dir, 'test')

    # Ensure directories exist (just in case)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 1. Get all file paths (exclude directories)
    # We look for .npy files directly in base_dir. 
    # We ignore files already inside train/val/test to prevent re-running issues.
    all_files = [f for f in glob.glob(os.path.join(base_dir, "*.npy")) if os.path.isfile(f)]

    if len(all_files) == 0:
        print("No .npy files found in root! Have you already moved them?")
        exit()

    # 2. Extract unique "Root IDs" (Dates)
    # filename format: data_20070504_093157_4.npy -> Root: 20070504_093157
    unique_roots = list(set([os.path.basename(f).split('data_')[1].rsplit('_', 1)[0] for f in all_files]))

    print(f"Total Patches Found: {len(all_files)}")
    print(f"Total Unique Rasters: {len(unique_roots)}")

    # 3. Split the ROOTS (Dates)
    # 80% Train, 10% Val, 10% Test
    train_roots, temp_roots = train_test_split(unique_roots, test_size=0.2, random_state=42)
    val_roots, test_roots = train_test_split(temp_roots, test_size=0.5, random_state=42)

    print(f"Train Rasters: {len(train_roots)}")
    print(f"Val Rasters:   {len(val_roots)}")
    print(f"Test Rasters:  {len(test_roots)}")

    # Convert lists to sets for O(1) lookup
    train_set = set(train_roots)
    val_set   = set(val_roots)
    test_set  = set(test_roots)

    # 4. Move Files
    print("\nMoving files...")
    count_train = 0
    count_val = 0
    count_test = 0

    for f_path in tqdm(all_files):
        filename = os.path.basename(f_path)
        
        # Extract root from filename
        root_id = filename.split('data_')[1].rsplit('_', 1)[0]
        
        if root_id in train_set:
            shutil.move(f_path, os.path.join(train_dir, filename))
            count_train += 1
        elif root_id in val_set:
            shutil.move(f_path, os.path.join(val_dir, filename))
            count_val += 1
        elif root_id in test_set:
            shutil.move(f_path, os.path.join(test_dir, filename))
            count_test += 1
        else:
            print(f"Warning: Could not assign {filename}")

    print("\nDone!")
    print(f"Moved {count_train} files to /train")
    print(f"Moved {count_val} files to /val")
    print(f"Moved {count_test} files to /test")
