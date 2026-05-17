import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch, glob, os, json
from slitless.measure import compare_ssim, nrmse
from slitless.common import outch_adjuster
from slitless.networks.unet import UNet
from torch.utils.data import DataLoader
from slitless.config import config
from slitless.data_loader import (BasicDataset, param_inv_transform, 
    meas_inv_transform, param_transform, meas_transform)
from slitless.plotting import barplot_group

def predict(net, meas):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(meas.shape)==3:
        meas = meas[np.newaxis]
    meas = torch.from_numpy(meas)
    meas = meas.to(device=device, dtype=torch.float)
    net = net.to(device)
    with torch.no_grad():
        pred = net(meas)
    pred = pred.squeeze().cpu().numpy()
    return pred

def net_loader(path):
    modpath = path + '/best_model.pth'
    json_path = os.path.join(path, 'checkpoint_metadata.json')

    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        start_filters = meta['num_filters']
        in_channels = meta['numdetectors']
        outch = meta['outch_type']
        out_channels = 3 if outch == 'all' else 1
        ksizes = meta['ksizes']
        bilinear = meta['bilinear']
        numlayers = len(ksizes)
        numlayers = 4 if numlayers == 1 else numlayers
    else:
        with open(path + '/summary.txt', 'r') as summary_text:
            lines = summary_text.readlines()
        parser = lambda key: [i for i in lines if key in i][0].split('= ')[-1].split(' \n')[0]
        start_filters = int(parser('Number of starting'))
        in_channels = int(parser('Number of Detectors'))
        outch = parser('Output Channels')
        out_channels = 3 if outch == 'all' else 1
        ksizes = eval(parser('Kernel Size'))
        bilinear = eval(parser('Bilinear Interpolation'))
        numlayers = len(ksizes)
        numlayers = 4 if numlayers == 1 else numlayers

    net = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        numlayers=numlayers,
        outch_type=outch,
        start_filters=start_filters,
        bilinear=bilinear,
        ksizes=ksizes,
        residual=False)
    net.load_state_dict(torch.load(modpath))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    return net

def plot_recons_gd(*,meas, truth, recon, savedir):
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    ssims = compare_ssim(truth=truth, estimate=recon)
    rmses = nrmse(truth=truth, estimate=recon, normalization=None)
    ssims = ssims.squeeze()
    rmses = rmses.squeeze()

    numim = truth.shape[0]

    matplotlib.use('Agg')
    for i in range(numim):
        fig, ax = plt.subplots(3,3, figsize=(13,7))
        im=ax[0,0].imshow(meas[i,0], cmap='hot')
        ax[0,0].set_title('Meas 0')
        fig.colorbar(im, ax=ax[0,0])
        im=ax[0,1].imshow(meas[i,1], cmap='hot')
        ax[0,1].set_title('Meas -1')
        fig.colorbar(im, ax=ax[0,1])
        im=ax[0,2].imshow(meas[i,2], cmap='hot')
        ax[0,2].set_title('Meas +1')
        fig.colorbar(im, ax=ax[0,2])
        im=ax[1,0].imshow(truth[i,0], cmap='hot')
        ax[1,0].set_title('True Intensity')
        fig.colorbar(im, ax=ax[1,0])
        im=ax[1,1].imshow(truth[i,1], cmap='seismic')
        ax[1,1].set_title('True Velocity')
        fig.colorbar(im, ax=ax[1,1])
        im=ax[1,2].imshow(truth[i,2], cmap='plasma')
        ax[1,2].set_title('True Linewidth')
        fig.colorbar(im, ax=ax[1,2])
        im=ax[2,0].imshow(recon[i,0], cmap='hot')
        ax[2,0].set_title(
            'Predicted Intensity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                ssims[i,0], rmses[i,0]
            )
        )
        fig.colorbar(im, ax=ax[2,0])
        im=ax[2,1].imshow(recon[i,1], cmap='seismic')
        ax[2,1].set_title(
            'Predicted Velocity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                ssims[i,1], rmses[i,1]
            )
        )
        fig.colorbar(im, ax=ax[2,1])
        im=ax[2,2].imshow(recon[i,2], cmap='plasma')
        ax[2,2].set_title(
            'Predicted Linewidth\n SSIM={:.3f} - RMSE={:.3f}'.format(
                ssims[i,2], rmses[i,2]
            )
        )
        fig.colorbar(im, ax=ax[2,2])
        plt.tight_layout()
        plt.savefig(savedir+f'recons_{i}.png', dpi=300)
        plt.close()
    plt.close('all')
    try:
        matplotlib.use('QtAgg')
    except:
        pass


def plot_recons(net, valloader, numim, savedir, denormalize=False):
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = next(iter(valloader))
    numim = min(numim, x.shape[0])
    x = x.to(device=device, dtype=torch.float)
    y = np.array(y.cpu())
    net.eval()
    with torch.no_grad():
        out = net(x)
    out = np.array(out.cpu())
    x = np.array(x.cpu())
    if denormalize:
        out = param_inv_transform(out, w_kms=True)
        y = param_inv_transform(y, w_kms=True)
        x = meas_inv_transform(x)
    if not hasattr(net, 'outch_type'):
        net.outch_type = 'all'

    y1 = outch_adjuster(out=None, true_out=y, outch_type=net.outch_type, action='crop')

    ssims = compare_ssim(truth=y1, estimate=out)
    rmses = nrmse(truth=y1, estimate=out, normalization=None)
    ssims = ssims.squeeze()
    rmses = rmses.squeeze()

    matplotlib.use('Agg')
    for i in range(numim):
        fig, ax = plt.subplots(3,3, figsize=(13,7))
        im=ax[0,0].imshow(x[i,0], cmap='hot')
        ax[0,0].set_title('Meas 0')
        fig.colorbar(im, ax=ax[0,0])
        im=ax[0,1].imshow(x[i,1], cmap='hot')
        ax[0,1].set_title('Meas -1')
        fig.colorbar(im, ax=ax[0,1])
        if x.shape[1]>2: # this is to avoid error if numdetectors==2
            im=ax[0,2].imshow(x[i,2], cmap='hot')
            ax[0,2].set_title('Meas +1')
            fig.colorbar(im, ax=ax[0,2])
        im=ax[1,0].imshow(y[i,0], cmap='hot')
        ax[1,0].set_title('True Intensity')
        fig.colorbar(im, ax=ax[1,0])
        im=ax[1,1].imshow(y[i,1], cmap='seismic')
        ax[1,1].set_title('True Velocity')
        fig.colorbar(im, ax=ax[1,1])
        im=ax[1,2].imshow(y[i,2], cmap='plasma')
        ax[1,2].set_title('True Linewidth')
        fig.colorbar(im, ax=ax[1,2])
        if net.outch_type == 'all':
            im=ax[2,0].imshow(out[i,0], cmap='hot')
            ax[2,0].set_title(
                'Predicted Intensity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i,0], rmses[i,0]
                )
            )
            fig.colorbar(im, ax=ax[2,0])
            im=ax[2,1].imshow(out[i,1], cmap='seismic')
            ax[2,1].set_title(
                'Predicted Velocity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i,1], rmses[i,1]
                )
            )
            fig.colorbar(im, ax=ax[2,1])
            im=ax[2,2].imshow(out[i,2], cmap='plasma')
            ax[2,2].set_title(
                'Predicted Linewidth\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i,2], rmses[i,2]
                )
            )
            fig.colorbar(im, ax=ax[2,2])
        elif net.outch_type == 'int':
            im=ax[2,0].imshow(out[i,0], cmap='hot')
            ax[2,0].set_title(
                'Predicted Intensity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i], rmses[i]
                )
            )
            fig.colorbar(im, ax=ax[2,0])
        elif net.outch_type == 'vel':
            im=ax[2,1].imshow(out[i,0], cmap='seismic')
            ax[2,1].set_title(
                'Predicted Velocity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i], rmses[i]
                )
            )
            fig.colorbar(im, ax=ax[2,1])
        elif net.outch_type == 'width':
            im=ax[2,2].imshow(out[i,0], cmap='plasma')
            ax[2,2].set_title(
                'Predicted Velocity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i], rmses[i]
                )
            )
            fig.colorbar(im, ax=ax[2,2])

        plt.tight_layout()
        plt.savefig(savedir+f'recons_{i}.png', dpi=300)
        plt.close()
    plt.close('all')
    try:
        matplotlib.use('QtAgg')
    except:
        pass

def stat_plotter(ssims, rmses, savedir):
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1,3, figsize=(12.6,4.8))
    ax[0].hist(ssims[:,0], bins=20)
    ax[0].set_title('Intensity SSIM\n Mean SSIM={:.3f}'.format(ssims[:,0].mean()))
    ax[0].set_xlabel('SSIM')
    ax[0].set_ylabel('Counts')
    ax[0].axvline(ssims[:,0].mean(), color='r')
    ax[1].hist(ssims[:,1], bins=20)
    ax[1].set_title('Velocity SSIM\n Mean SSIM={:.3f}'.format(ssims[:,1].mean()))
    ax[1].set_xlabel('SSIM')
    ax[1].set_ylabel('Counts')
    ax[1].axvline(ssims[:,1].mean(), color='r')
    ax[2].hist(ssims[:,2], bins=20)
    ax[2].set_title('Linewidth SSIM\n Mean SSIM={:.3f}'.format(ssims[:,2].mean()))
    ax[2].set_xlabel('SSIM')
    ax[2].set_ylabel('Counts')
    ax[2].axvline(ssims[:,2].mean(), color='r')
    plt.tight_layout()
    plt.savefig(savedir+'ssim_stats.png', dpi=300)

    fig, ax = plt.subplots(1,3, figsize=(12.6,4.8))
    ax[0].hist(rmses[:,0], bins=20)
    ax[0].set_title('Intensity RMSE\n Mean RMSE={:.3f}'.format(rmses[:,0].mean()))
    ax[0].set_xlabel('RMSE')
    ax[0].set_ylabel('Counts')
    ax[0].axvline(rmses[:,0].mean(), color='r')
    ax[1].hist(rmses[:,1], bins=20)
    ax[1].set_title('Velocity RMSE\n Mean RMSE={:.3f}'.format(rmses[:,1].mean()))
    ax[1].set_xlabel('RMSE')
    ax[1].set_ylabel('Counts')
    ax[1].axvline(rmses[:,1].mean(), color='r')
    ax[2].hist(rmses[:,2], bins=20)
    ax[2].set_title('Linewidth RMSE\n Mean RMSE={:.3f}'.format(rmses[:,2].mean()))
    ax[2].set_xlabel('RMSE')
    ax[2].set_ylabel('Counts')
    ax[2].axvline(rmses[:,2].mean(), color='r')
    plt.tight_layout()
    plt.savefig(savedir+'rmse_stats.png', dpi=300)
    plt.close('all')
    try:
        matplotlib.use('QtAgg')
    except:
        pass

def joint_plotter(truth, recon, savedir):
    est_bias = np.mean(recon - truth, axis=1) 
    est_std = np.std(recon - truth, axis=1)

    matplotlib.use('Agg')
    # fg=sns.jointplot(x=truth[0], y=recon[0]-truth[0], kind='hex', gridsize=100)
    fg=sns.jointplot(x=truth[0], y=recon[0]-truth[0], xlim=[0,5e3], ylim=[-2e2,2e2], kind='hex', gridsize=100)
    fg.figure.suptitle('Intensity Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[0], est_std[0]))
    fg.set_axis_labels('Intensity', 'Intensity Error')
    fg.figure.tight_layout()
    plt.grid(which='both', axis='both')
    plt.savefig(savedir+'intensity_stats.png', dpi=300)
    # fg=sns.jointplot(x=truth[1], y=recon[1]-truth[1], kind='hex', gridsize=100)
    fg=sns.jointplot(x=truth[1], y=recon[1]-truth[1], kind='hex', xlim=[-15,15], ylim=[-15,15], gridsize=300)
    fg.figure.suptitle('Velocity Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[1], est_std[1]))
    fg.set_axis_labels('Velocity', 'Velocity Error')
    fg.figure.tight_layout()
    plt.grid(which='both', axis='both')
    plt.savefig(savedir+'velocity_stats.png', dpi=300)
    # fg=sns.jointplot(x=truth[2], y=recon[2]-truth[2], kind='hex', gridsize=100)
    fg=sns.jointplot(x=truth[2], y=recon[2]-truth[2], xlim=[35,60], ylim=[-8,8], kind='hex', gridsize=100)
    fg.figure.suptitle('Linewidth Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[2], est_std[2]))
    fg.set_axis_labels('Linewidth', 'Linewidth Error')
    fg.figure.tight_layout()
    plt.grid(which='both', axis='both')
    plt.savefig(savedir+'linewidth_stats.png', dpi=300)

    # Cross dependence of vel&width errors on intensity
    fg=sns.jointplot(x=truth[0], y=recon[1]-truth[1], xlim=[0,5e3], ylim=[-8,8], kind='hex', gridsize=100)
    # fg=sns.jointplot(x=truth[0], y=recon[1]-truth[1], kind='hex', ylim=[-0.4,0.4], gridsize=100)
    fg.figure.suptitle('Velocity Error vs Intensity\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[1], est_std[1]))
    fg.set_axis_labels('Intensity', 'Velocity Error')
    fg.figure.tight_layout()
    plt.savefig(savedir+'velocity_stats_vs_inten.png', dpi=300)
    fg=sns.jointplot(x=truth[0], y=recon[2]-truth[2], xlim=[0,5e3], ylim=[-8,8], kind='hex', gridsize=100)
    # fg=sns.jointplot(x=truth[0], y=recon[2]-truth[2], kind='hex', ylim=[-0.5,0.5], gridsize=100)
    fg.figure.suptitle('Linewidth Error vs Intensity\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[2], est_std[2]))
    fg.set_axis_labels('Intensity', 'Linewidth Error')
    fg.figure.tight_layout()
    plt.savefig(savedir+'linewidth_stats_vs_inten.png', dpi=300)

    plt.close('all')
    try:
        matplotlib.use('QtAgg')
    except:
        pass

def plot_val_stats(net, valloader, savedir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()

    ssims=[]
    rmses=[]
    yvec=[]
    outvec=[]
    for i, data in enumerate(valloader):
        # get the inputs
        inputs = data[0].to(device=device, dtype=torch.float)
        true_outputs = np.array(data[1].cpu())
        if not hasattr(net, 'outch_type'):
            net.outch_type = 'all'

        y1 = outch_adjuster(out=None, true_out=true_outputs, outch_type=net.outch_type, action='crop')
        TITLE_MAP = {'int': 'Intensity', 'vel': 'Velocity', 'width': 'Linewidth'}
        title_str = TITLE_MAP.get(net.outch_type, '')

        with torch.no_grad():
            outputs = net(inputs)
            outputs = np.array(outputs.cpu())
            outputs = param_inv_transform(outputs, w_kms=True)
            y1 = param_inv_transform(y1, w_kms=True)
            inputs = meas_inv_transform(inputs)
            ssim0 = compare_ssim(truth=y1, estimate=outputs)
            rmse0 = nrmse(truth=y1, estimate=outputs, normalization=None)
            ssims.extend(ssim0.squeeze())
            rmses.extend(rmse0.squeeze())
            if i*valloader.batch_size<10000:
                if net.outch_type == 'all':
                    yvec.extend(y1.transpose(1,0,2,3).reshape(3,-1).transpose(1,0))
                    outvec.extend(outputs.transpose(1,0,2,3).reshape(3,-1).transpose(1,0))
                else:
                    yvec.extend(y1.transpose(1,0,2,3).reshape(1,-1).transpose(1,0))
                    outvec.extend(outputs.transpose(1,0,2,3).reshape(1,-1).transpose(1,0))

    ssims = np.array(ssims)
    rmses = np.array(rmses)
    yvec = np.array(yvec).transpose(1,0)
    outvec = np.array(outvec).transpose(1,0)
    est_bias = np.mean(outvec - yvec, axis=1) 
    est_std = np.std(outvec - yvec, axis=1)

    if net.outch_type == 'all':
        stat_plotter(ssims, rmses, savedir)
    else:
        plt.figure()
        plt.hist(ssims, bins=20)
        plt.title('{} SSIM\n Mean SSIM={:.3f}'.format(title_str, ssims.mean()))
        plt.xlabel('SSIM')
        plt.ylabel('Counts')
        plt.axvline(ssims.mean(), color='r')
        plt.tight_layout()
        plt.savefig(savedir+'ssim_stats.png', dpi=300)

        plt.figure()
        plt.hist(rmses, bins=20)
        plt.title('{} RMSE\n Mean RMSE={:.3f}'.format(title_str, rmses.mean()))
        plt.xlabel('RMSE')
        plt.ylabel('Counts')
        plt.axvline(rmses.mean(), color='r')
        plt.tight_layout()
        plt.savefig(savedir+'rmse_stats.png', dpi=300)

    if net.outch_type == 'all':
        joint_plotter(yvec, outvec, savedir)

    if net.outch_type == 'int':
        # fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', ylim=[-0.1,0.1], gridsize=100)
        fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', gridsize=100)
        fg.fig.suptitle('Intensity Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[0], est_std[0]))
        fg.set_axis_labels('Intensity', 'Intensity Error')
        fg.fig.tight_layout()
        plt.savefig(savedir+'intensity_stats.png', dpi=300)

    if net.outch_type == 'vel':
        # fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', ylim=[-0.4,0.4], gridsize=100)
        fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', gridsize=100)
        fg.fig.suptitle('Velocity Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[0], est_std[0]))
        fg.set_axis_labels('Velocity', 'Velocity Error')
        fg.fig.tight_layout()
        plt.savefig(savedir+'velocity_stats.png', dpi=300)

    if net.outch_type == 'width':
        # fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', ylim=[-0.5,0.5], gridsize=100)
        fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', gridsize=100)
        fg.fig.suptitle('Linewidth Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[0], est_std[0]))
        fg.set_axis_labels('Linewidth', 'Linewidth Error')
        fg.fig.tight_layout()
        plt.savefig(savedir+'linewidth_stats.png', dpi=300)

    return ssims, rmses, yvec, outvec

def eval_snrlist(dbsnr_list, noise_model, fold, data_dir, net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device=device)

    dsetter = BasicDataset
    # if fold=='test':
    #     dsetter = BasicDataset
    # else:
    #     dsetter = OntheflyDataset

    ssims_l = []
    rmses_l = []
    for dbsnr in dbsnr_list:
        print(dbsnr)
        dset = dsetter(data_dir=data_dir, noise_model=noise_model, fold=fold, dbsnr=dbsnr, numdetectors=net.in_channels)
        dloader = DataLoader(dset, batch_size=32, shuffle=True, num_workers=8)

        ssims=[]
        rmses=[]
        for i, data in enumerate(dloader):
            # get the inputs
            inputs = data[0].to(device=device, dtype=torch.float)
            true_outputs = np.array(data[1].cpu())
            if not hasattr(net, 'outch_type'):
                net.outch_type = 'all'

            y1 = outch_adjuster(out=None, true_out=true_outputs, outch_type=net.outch_type, action='crop')

            with torch.no_grad():
                outputs = net(inputs)
                outputs = np.array(outputs.cpu())
                ssim0 = compare_ssim(truth=y1, estimate=outputs)
                rmse0 = nrmse(truth=y1, estimate=outputs, normalization=None)
                ssims.extend(ssim0)
                rmses.extend(rmse0)
        
        ssims_l.append(ssims)
        rmses_l.append(rmses)
    return np.array(ssims_l), np.array(rmses_l)

def meanest_errcalc(trainmeans, dataloader):
    rmse = torch.zeros(3)
    mae = torch.zeros(3)
    ssim = torch.zeros(3)
    estt = trainmeans[None,:,None,None]*torch.ones_like(next(iter(dataloader))[1])
    for data in dataloader:
        estt = trainmeans[None,:,None,None]*torch.ones_like(data[1])
        rmse += torch.mean((data[1]-estt)**2, dim=(0,2,3))
        mae += torch.mean(torch.abs(data[1]-estt), dim=(0,2,3))
        ssim += compare_ssim(truth=data[1], estimate=estt).mean(axis=0)
    ssim /= len(dataloader)
    rmse /= len(dataloader)
    rmse = torch.sqrt(rmse)
    mae /= len(dataloader)
    return ssim, rmse, mae

if __name__ == '__main__':
    # foldname0 = '2023_01_19__17_18_44_NF_64_BS_4_LR_0.0002_EP_200_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_35_dssize_full'
    foldname0 = '2026_01_24__15_28_17_NF_64_BS_4_LR_0.0002_EP_200_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_100_None_K_3_dssize_full'
    foldpath = glob.glob('../results/saved/'+foldname0)[0]+'/'
    net = net_loader(foldpath)
    dsetname='eistest64'
    # dataset_path = glob.glob(f'../../data/eis_data/{dsetname}/')[0]
    dataset_path = str(config.train_data_dir.parent.parent / 'dset_v4' / 'data')
    valset = BasicDataset(data_dir=dataset_path, transform=meas_transform, target_transform=param_transform, fold='val', dbsnr=None, noise_model=None, numdetectors=3)
    dataloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=8)

    savedir = foldpath
    # if not os.path.exists(savedir):
    #     os.mkdir(savedir)

    ssims, rmses, yvec, outvec = plot_val_stats(net, dataloader, savedir)