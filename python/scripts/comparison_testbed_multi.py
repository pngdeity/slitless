from slitless.forward import Source, Imager, forward_op, datacube_generator
from slitless.plotting import uiuc_im
from slitless.recon import (smart, smart2, smart2_twostage, grad_descent_solver, scipy_solver, scipy_solver_parallel, scipy_solver_parallel2, tomoinv,
    Reconstructor, Reconstructor_Multi, nn_solver, diffusion_solver)
import numpy as np
import datetime, pickle, shutil, os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from slitless.data_loader import param_inv_transform
from slitless.eistools import meas_boundary_corrector
from slitless.config import config
import matplotlib
# matplotlib.use('Agg')

path_data = str(config.data_root / 'datasets/baseline')
# data='apj19_20x20.npy' # 20x20 APJ2019 image
# data='eis_5_64x64.npy' # 64x64 EIS images
# data='diff_samples.npy' # 5 of 64x64 Diffusion generated images
# data = 'eis_train_5_64x64.npy' # 5 of 64x64 EIS dataset train images
data_file='eis_test_100_dsetv5.npy' # 64x64 EIS images
# data='eis_train_5_dsetv4.npy' # 64x64 EIS images
# data='eis_test_5_dsetv4.npy' # 64x64 EIS images
# data='eis_val_5_dsetv4_old.npy' # 64x64 EIS images

# # param4dar = uiuc_im()
# param4dar = np.load
# (path_data+data_)[[1]]
# if len(param4dar.shape)<4:
#     param4dar = param4dar[np.newaxis]
# source_pix = True
# intenscaling=False
# meas4dar=None

# Loading script for dset_v4 data
data = np.load(path_data+data_file, allow_pickle=True).item()
# param4dar, meas4dar = data['param3d'], data['meas']
# param4dar, meas4dar = data['param3d'], data['meas']
param4dar, meas4dar = data['param3d'], data['meas_damped']
# param4dar, meas4dar = data['param3d'][[3]], data['meas_damped'][[3]]
source_pix = False
intenscaling = False
# meas4dar=None

savepath = str(config.results_root / 'recons')
save = False
M = param4dar.shape[-1]
numdetectors = 3
dbsnr = 10
noise_model=None # Noise-free measurements
# noise_model='poisson'
# noise_model='gaussian'
if meas4dar is not None:
    meas4dar = meas4dar[:,:numdetectors]

Imgr = Imager(pixelated=True, dbsnr=dbsnr, avg_count=dbsnr**2, noise_model=noise_model, 
# Imgr = Imager(pixelated=True, mask=mask, dbsnr=dbsnr, max_count=dbsnr**2/0.9, noise_model=noise_model, 
    spectral_orders=[0,-1,1,-2,2][:numdetectors])

# sr = Source(param3d=param4dar[0],pix=True)
# m=Imgr.get_measurements(sources=sr)
# Imgr.plot('poisson_np: {}'.format(poisson_sn))
# if meas4dar is not None:
#     meas4dar = meas_boundary_corrector(Imgr, meas4dar, param4dar)

# # SCIPY
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     meas4dar=meas4dar,
#     pix=source_pix,
#     solver=scipy_solver_parallel,
#     intenscaling=intenscaling,
#     DATA_FIDELITY='L2',
#     OPTIMIZER='L-BFGS-B',
#     maxiter=10000,
#     lam_i=1e-4,
#     lam_v=1.5e4,
#     lam_w=2e9
# )

# # SCIPY
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     meas4dar=meas4dar,
#     pix=source_pix,
#     solver=scipy_solver_parallel2,
#     intenscaling=intenscaling,
#     DATA_FIDELITY='L2',
#     OPTIMIZER='L-BFGS-B',
#     maxiter=10000,
#     lam_i=1e4,
#     lam_v=1.2e-1,
#     lam_w=2.6e-2,
#     frac1=0.8620,
#     frac2=0.0521,
#     frac_bg=0.0860,
#     cent1=195.11723,
#     wid1=0.02981,
#     cent2=195.17723,
#     wid2=0.02981,
#     bg_shape_norm=[0.04762] * 21
# )

# Tomoinv
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     pix=source_pix,
#     solver=tomoinv,
#     lam=1e-2,
#     stepsize=3e-2,
#     numiter=100,
#     proj='positivity',
#     data_step='grad',
#     positivity=True
# )

# # 2D GD
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     pix=source_pix,
#     solver=grad_descent_solver,
#     maxiter=1000,
#     lam_i=1e-8,
#     # lam_v=0.0001,
#     # lam_w=4.6e-3,
#     lam_v=0.000278,
#     lam_w=6e-4,
#     LR=5e-2
# )

# U-Net
Rec = Reconstructor_Multi(
    imager=Imgr,
    meas4dar=meas4dar,
    param4dar=param4dar,
    pix=source_pix,
    solver=nn_solver,
    intenscaling=intenscaling,
    # model_path='dbsnr_50_poisson_K_3_dssize_full',
    model_path='2026_05_15__01_06_51_NF_64_BS_4_LR_0.0002_EP_200_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_30_None_K_3_eis_v5'
    # model_path='2026_05_11__17_26_39_NF_64_BS_4_LR_0.0002_EP_400_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_100_None_K_3_eis_v5'
    # model_path='dbsnr_15_poisson_K_3_eis_v4'
)

# # Diffusion DPS
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     pix=source_pix,
#     solver=diffusion_solver,
#     model_path='model-10.pt',
#     grad_scale=[1,1,1],
#     num_samples=10
# )

# # SMART
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     meas4dar=meas4dar,
#     param4dar=param4dar,
#     pix=source_pix,
#     solver=smart2_twostage,
#     fitter='mpfit',
#     intenscaling=intenscaling,
#     psi=0.2,
#     maxouter=5,
#     maxinner=20,
#     prior_weight=1,
#     cent1=-1.13*(195.11794/299792.458)+195.11803,
#     wid1=42.74*(195.11794/299792.458),
#     wid2=42.74*(195.11794/299792.458),
#     n_jobs=-1
# )

recons = Rec.solve(num_realizations=1)
# Rec.recons[0].plot_loss()
fig, ax = recons[0].plot(compare=True, title=f'{Rec.solver.__name__}')
fig, ax = recons[1].plot(compare=True, title=f'{Rec.solver.__name__}')
fig, ax = recons[2].plot(compare=True, title=f'{Rec.solver.__name__}')
fig, ax = recons[3].plot(compare=True, title=f'{Rec.solver.__name__}')
fig, ax = recons[4].plot(compare=True, title=f'{Rec.solver.__name__}')
# print('mask: {}'.format(mask[:2,:2]))
print('Solver: {}'.format(Rec.solver.__name__))
print('Solver Params: {}'.format(Rec.solver_params))
print('Recon Time Avg: {:.2f} s'.format(Rec.times.mean()))
# print('RMSE_phy Avg (per Img): {}'.format(Rec.rmse_phy.mean(axis=1)))
print('RMSE_phy Avg: {}'.format(Rec.rmse_phy.mean(axis=(0,1))))
print('MAE_phy Avg: {}'.format(Rec.mae_phy.mean(axis=(0,1))))
# print('Bias_phy Avg (per Img): {}'.format(Rec.bias_phy.mean(axis=1)))
print('Bias_phy Avg: {}'.format(Rec.bias_phy.mean(axis=(0,1))))
# print('RMSE_pix Avg (per Img): {}'.format(Rec.rmse_pix.mean(axis=1)))
# print('RMSE_pix Avg: {}'.format(Rec.rmse_pix.mean(axis=(0,1))))
# print('SSIMs: {}'.format(recons.ssim))
# print('SSIM Avg: {}'.format(recons.ssim.mean(axis=0)))

if save==True:
    now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    name = (f'{now}_{Rec.solver.__name__}_{data_file[:-4]}_K_{numdetectors}_{noise_model}_dbsnr_{dbsnr}')
    savedir = savepath+name
    os.mkdir(savedir)
    recon_summary = [
    '############## Recon Parameters ############## \n',
    # 'mask: {} \n'.format(mask[:2,:2]),
    'Solver: {} \n'.format(Rec.solver.__name__),
    'Num Detectors: {} \n'.format(numdetectors),
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

    rec_array_pix = []
    rec_array_phy = []
    truth_array_pix = []
    truth_array_phy = []
    for i in range(len(Rec.recons)):
        fig, ax = recons[i].plot(compare=True, title=f'{Rec.solver.__name__}')
        fig.savefig(savedir+f'/recon_{i}.png')
        rec_array_pix.append(Rec.recons[i].recon)
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
        plt.show()
        return fig, ax
    fig, ax = hist_plotter(diff_phy, 'phy')
    fig.savefig(savedir+'/error_hist_phy.png')
    fig, ax = hist_plotter(diff_pix, 'pix')
    fig.savefig(savedir+'/error_hist_pix.png')

    with open(f'{savedir}/summary.txt', 'w') as file:
        for line in recon_summary:
            file.write(line)

    with open(f'{savedir}/Rec.pickle', 'wb') as file:
        pickle.dump(Rec, file)

# from slitless.recon import comparison_test_multi
# path_data = str(config.data_root / 'datasets/baseline')
# data = 'eis_train_5_64x64.npy' # 5 of 64x64 EIS dataset train images
# savepath = str(config.results_root / 'recons')
# Rec_result = comparison_test_multi(path_data, data, savepath, single_param4dar=True, save=False, numdetectors=3, dbsnr=50, 
#                           noise_model='poisson', solver='diffusion', model_path='model-10.pt', grad_scale=[1,1,1], num_samples=10