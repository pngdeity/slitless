import logging, glob, os
from os import listdir
from os.path import splitext

import numpy as np
import torch, time
from slitless.config import config
from slitless.forward import add_noise, forward_op, forward_op_torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool

WAVELENGTH = config.wavelength
SPEEDOFLIGHT = config.speed_of_light

_stats_cache = None

def _get_stats():
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = np.load(str(config.norm_stats_path), allow_pickle=True).item()
    return _stats_cache

def meas_transform(meas, stats=None):
    if stats is None:
        stats = _get_stats()
    return meas / config.intensity_scale

def meas_inv_transform(meas, stats=None):
    if stats is None:
        stats = _get_stats()
    return meas * config.intensity_scale

def param_transform(params, stats=None):
    if stats is None:
        stats = _get_stats()
    params[0] /= config.intensity_scale
    params[1] = (params[1] - stats['vel_mean']) / stats['vel_std']
    params[2] = (params[2] - stats['width_mean']) / stats['width_std']
    return params

def param_inv_transform(params, w_kms=False, stats=None):
    if stats is None:
        stats = _get_stats()
    params[...,0,:,:] *= config.intensity_scale
    params[...,1,:,:] = stats['vel_std'] * params[...,1,:,:] + stats['vel_mean']
    params[...,2,:,:] = stats['width_std'] * params[...,2,:,:] + stats['width_mean']
    if w_kms:
        params[...,2,:,:] *= SPEEDOFLIGHT / WAVELENGTH
    return params

class BasicDataset(Dataset):
    def __init__(self, data_dir, fold='train', transform=None,
        target_transform=None, dbsnr=None, noise_model=None, numdetectors=3):
        self.data_dir = data_dir
        self.data = []
        self.train = False
        self.val = False
        self.transform = transform
        self.target_transform = target_transform
        self.dbsnr = dbsnr
        self.noise_model = noise_model
        self.numdetectors = numdetectors

        if fold == 'train':
            self.train = True
            self.task_dir = os.path.join(data_dir, 'train')
        elif fold == 'val':
            self.val = True
            self.task_dir = os.path.join(data_dir, 'val')
        elif fold == 'test':
            self.test = True
            self.task_dir = os.path.join(data_dir, 'test')

        self.files = glob.glob(self.task_dir+'/data*.npy')
        self.files.sort()
        for file in self.files:
            data = np.load(file, allow_pickle=True).item()
            params = np.stack([data['int'], data['vel'], data['width']])
            meas = np.stack([data['meas_0'], data['meas_-1'], data['meas_1'], data['meas_-2'], data['meas_2']])[:numdetectors]
            self.data.append((meas, params))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        meas, params = self.data[idx]

        if self.transform is not None:
            meas = self.transform(meas)

        if self.target_transform is not None:
            params = self.target_transform(params)

        meas = add_noise(meas, dbsnr=self.dbsnr, no_noise=self.dbsnr==None,
            noise_model=self.noise_model)

        return meas, params

class OntheflyDataset(Dataset):
    def __init__(self, data_dir, fold='train', transform=None,
        target_transform=None, dbsnr=None, trpart=None):
        self.data_dir = data_dir
        self.data = []
        self.train = False
        self.val = False
        self.transform = transform
        self.target_transform = target_transform
        self.dbsnr = dbsnr
        self.trpart = trpart

        if fold == 'train':
            self.train = True
        elif fold == 'val':
            self.val = True

        self.task_dir = os.path.join(self.data_dir, fold)

        if trpart is None:
            file = glob.glob(self.task_dir+'/otf*npy')[0]
        else:
            file = glob.glob(self.task_dir+'/otf_p{}*npy'.format(trpart))[0]
        data = np.load(file)
        intens, vels, widths = data.reshape(3,-1,64,64)

        vel_max = 2 # pixels
        width_max = 2.5 # pixels
        width_min = 1 # pixels

        v0, v1 = (np.random.uniform(0, vel_max, (2,len(vels))) * [[-1],[1]])[:,:,None,None]
        w0, w1 = np.sort(np.random.uniform(width_min, width_max, (2,len(vels))), axis=0)[:,:,None,None]
        
        vels = vels * (v1 - v0) + v0
        widths = widths * (w1 - w0) + w0

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda')
        intens = torch.from_numpy(intens).to(device=device, dtype=torch.float)
        vels = torch.from_numpy(vels).to(device=device, dtype=torch.float)
        widths = torch.from_numpy(widths).to(device=device, dtype=torch.float)

        meas = []
        ind = 1000

        for i in range(len(intens) // ind):
            meas.extend(forward_op_torch(
                true_intensity=intens[i*ind:(i+1)*ind],
                true_doppler=vels[i*ind:(i+1)*ind],
                true_linewidth=widths[i*ind:(i+1)*ind],
                spectral_orders=[0,-1,1],
                pixelated=True,
                device=device
            ).cpu().numpy())

        self.data = np.concatenate((meas,intens.cpu()[:,None],vels.cpu()[:,None],widths.cpu()[:,None]), axis=1)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meas = self.data[idx, :3]
        params = self.data[idx, 3:]

        meas = add_noise(meas, dbsnr=self.dbsnr, no_noise=self.dbsnr==None, 
            model='Gaussian')

        if self.transform is not None:
            meas = self.transform(meas)

        if self.target_transform is not None:
            params = self.target_transform(params)

        return meas, params

def dloadertesting():
    dataset_path = glob.glob('../../data/datasets/dset6*')[0]
    t0 = time.time()
    trainset = OntheflyDataset(data_dir=dataset_path, fold='train', dbsnr=35)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=16)
    print('Time Dataset Onthefly: {}'.format(time.time()-t0))
    t0 = time.time()
    for i,data in enumerate(trainloader):
        # tada = data
        if i==2000:
            break
    print('Time Loop Onthefly: {}'.format(time.time()-t0))

    dataset_path = glob.glob('../../data/datasets/dset5*')[0]
    t0 = time.time()
    trainset = BasicDataset(data_dir=dataset_path, fold='train', dbsnr=35)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=16)
    print('Time Dataset Basic: {}'.format(time.time()-t0))

    t0 = time.time()
    for i,data in enumerate(trainloader):
        # tada = data
        if i==2000:
            break
    print('Time Loop Basic: {}'.format(time.time()-t0))

def data_rewrite(file):
    name = file.split('/')[-1]
    out_name = 'data/train2/{}'.format(name)
    data = np.load(file, allow_pickle=True).item()
    im_starry = data['image_stars']
    im_clean = data['image_ori']
    im_starry = np.where(np.isnan(im_starry), im_clean, im_starry)
    array = np.stack((im_starry,im_clean))
    np.save(out_name, array)

def data_rewriter_par():
    files = glob.glob('data/train/*')
    pool = Pool()
    pool.map(data_rewrite, files)

# def eis_data_plotter(folder):