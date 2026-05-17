import torch
import sys, os, logging, time, glob
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import datetime

from slitless.config import config
from slitless.data_loader import (BasicDataset, OntheflyDataset,
    meas_inv_transform, meas_transform, param_transform)
from slitless.networks.unet import UNet
from slitless.measure import nrmse, nmse_torch, combine_losses, cycle_loss, compare_ssim
from slitless.common import outch_adjuster
import numpy as np
import matplotlib.pyplot as plt
from slitless.evaluate import plot_recons, plot_val_stats, eval_snrlist, meanest_errcalc
from slitless.plotting import barplot_group

def train_net(net,
            device,
            trainloader,
            otf,
            valloader,
            epochs,
            optimizer,
            criterion,
            path
):

    train_loss_over_epochs = []
    val_loss_over_epochs = []
    train_ssim_over_epochs = []
    val_ssim_over_epochs = []
    train_rmse_over_epochs = []
    val_rmse_over_epochs = []
    best_valloss = 1e6
    modnum = 5 if otf is not None else 1

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

        # if onthefly dataloading is active, then read the parameters from otf
        # and create the next trainset 
        if otf is True:
            trainset = OntheflyDataset(data_dir=trainloader.dataset.data_dir, fold='train', dbsnr=trainloader.dataset.dbsnr, trpart=epoch%5+1)
            trainloader = DataLoader(trainset, batch_size=trainloader.batch_size, shuffle=True, num_workers=trainloader.num_workers)

        net.train()
        running_loss = 0.0
        running_rmse_train = 0.0
        running_ssim_train = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs = data[0].to(device=device, dtype=torch.float)
            true_outputs = data[1].to(device=device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(truth=true_outputs, out=outputs, meas=inputs)
            # print('Loss Calculated')
            loss.backward()
            # print('Backward Propagated')
            optimizer.step()
            # print('Optimizer Updated')

            # print statistics
            running_loss += loss.item()

            outputs = outch_adjuster(out=outputs, true_out=true_outputs, outch_type=net.outch_type, action='extend')
            with torch.no_grad():
                running_rmse_train += torch.mean((true_outputs-outputs)**2, dim=(0,2,3)).cpu().numpy()
                running_ssim_train += compare_ssim(truth=true_outputs.cpu().numpy(), estimate=outputs.detach().cpu().numpy()).mean(axis=0)

        # Normalizing the loss by the total number of train batches
        running_loss/=len(trainloader)
        running_rmse_train/=len(trainloader)
        running_ssim_train/=len(trainloader)
        running_rmse_train=np.sqrt(running_rmse_train)
        logging.info('[%d] Train loss: %.7f' % (epoch + 1, running_loss))

        if (epoch+1)%modnum==0:
            net.eval()
            running_valloss = 0.0
            running_rmse_val = 0.0
            running_ssim_val = 0.0
            for i, data in enumerate(valloader):
                # get the inputs
                inputs = data[0].to(device=device, dtype=torch.float)
                true_outputs = data[1].to(device=device, dtype=torch.float)

                with torch.no_grad():
                    outputs = net(inputs)
                    loss = criterion(truth=true_outputs, out=outputs, meas=inputs)

                    outputs = outch_adjuster(out=outputs, true_out=true_outputs, outch_type=net.outch_type, action='extend')
                    running_valloss += loss.item()
                    running_rmse_val += torch.mean((true_outputs-outputs)**2, dim=(0,2,3)).cpu().numpy()
                    running_ssim_val += compare_ssim(truth=true_outputs.cpu().numpy(), estimate=outputs.cpu().numpy()).mean(axis=0)

            running_valloss/=len(valloader)
            running_rmse_val/=len(valloader)
            running_ssim_val/=len(valloader)
            running_rmse_val=np.sqrt(running_rmse_val)
            logging.info('Validation loss: %.7f' % (running_valloss))
            logging.info(['Train SSIM: ', ['{:.3f}'.format(ss) for ss in running_ssim_train]])
            logging.info(['Val SSIM: ', ['{:.3f}'.format(ss) for ss in running_ssim_val]])
            logging.info(['Train RMSE: ', ['{:.4f}'.format(ss) for ss in running_rmse_train]])
            logging.info(['Val RMSE: ', ['{:.4f}'.format(ss) for ss in running_rmse_val]])

            if running_valloss < best_valloss:
                best_valloss = running_valloss*1
                torch.save(net.state_dict(), f'../results/saved/{name}/best_model.pth')

            train_loss_over_epochs.append(running_loss)
            val_loss_over_epochs.append(running_valloss)
            train_ssim_over_epochs.append(running_ssim_train)
            val_ssim_over_epochs.append(running_ssim_val)
            train_rmse_over_epochs.append(running_rmse_train)
            val_rmse_over_epochs.append(running_rmse_val)

    return (train_loss_over_epochs, val_loss_over_epochs, 
    np.array(train_ssim_over_epochs), np.array(val_ssim_over_epochs), 
    np.array(train_rmse_over_epochs), np.array(val_rmse_over_epochs), net)

if __name__ == '__main__':
    # ---------
    numdetectors = config.numdetectors
    NUM_FILT = 64
    numlayers = 4
    LR = config.learning_rate
    # LR= 0.01
    EPOCHS = 100
    BATCH_SIZE = config.batch_size
    BILINEAR = config.bilinear
    ksizes = [(3,1)]
    OPTIMIZER = config.optimizer_type.upper()
    # LOSS = 'MSE'
    LOSS = 'NMSE'
    CYC_LOSS = False
    cyc_lam = 1
    CYC_ONLY = False
    LOSS = 'CYCLE_ONLY' if CYC_ONLY else LOSS
    OUTCH = config.outch_type
    out_channels = 3 if OUTCH=='all' else 1
    LOAD = True
    otf = None # on the fly trainset generation 
    loaded_model_path = '../results/saved/2026_05_11__17_26_39_NF_64_BS_4_LR_0.0002_EP_400_KSIZE_(3, 1)_NMSE_LOSS_ADAM_all_dbsnr_100_None_K_3_eis_v5/best_model.pth'
    # dataset_path = glob.glob('../../data/datasets/dset8_imagenet_50000/')[0]
    dataset_path = str(config.train_data_dir)
    testset_path = str(config.train_data_dir)
    dbsnr = 30
    noise_model = config.noise_model
    # noise_model = None

    now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    name = f'{now}_NF_{NUM_FILT}_BS_{BATCH_SIZE}_LR_{LR}_EP_{EPOCHS}_KSIZE_{str(ksizes[0])}_{LOSS}_LOSS_{OPTIMIZER}_{OUTCH}_dbsnr_{dbsnr}_{noise_model}_K_{numdetectors}'
    if (not CYC_ONLY) & CYC_LOSS:
        name += f'_CYC_LOSS_lam_{cyc_lam}'
    name += '_eis_v5'
    os.mkdir('../results/saved/'+name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [] # clean up the previous handlers to avoid problems
    fh = logging.FileHandler(f'../results/saved/{name}/output.log')
    formatter = logging.Formatter('%(asctime)s; %(message)s','%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    trainset = BasicDataset(data_dir=dataset_path, transform=meas_transform, target_transform=param_transform, fold='train', dbsnr=dbsnr, noise_model=noise_model, numdetectors=numdetectors)
    valset = BasicDataset(data_dir=dataset_path, transform=meas_transform, target_transform=param_transform, fold='val', dbsnr=dbsnr, noise_model=noise_model, numdetectors=numdetectors)
    testset = BasicDataset(data_dir=testset_path, transform=meas_transform, target_transform=param_transform, fold='test', dbsnr=dbsnr, noise_model=noise_model, numdetectors=numdetectors)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)

    # trainset = OntheflyDataset(data_dir=dataset_path, fold='train', dbsnr=dbsnr)
    # trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    # valset = OntheflyDataset(data_dir=dataset_path, fold='val', dbsnr=dbsnr)
    # valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=8)

    trainmeans = torch.zeros(3)
    for i, data in enumerate(trainloader):
        trainmeans += data[1].mean(dim=(0,2,3))
    trainmeans /= len(trainloader)

    ssim_cnst_train, rmse_cnst_train, mae_cnst_train = meanest_errcalc(
        trainmeans, trainloader)
    ssim_cnst_val, rmse_cnst_val, mae_cnst_val = meanest_errcalc(
        trainmeans, valloader)
    ssim_cnst_test, rmse_cnst_test, mae_cnst_test = meanest_errcalc(
        trainmeans, testloader)

    net = UNet(
        in_channels=numdetectors,
        out_channels=out_channels,
        numlayers=numlayers,
        outch_type=OUTCH,
        start_filters=NUM_FILT,
        bilinear=BILINEAR,
        ksizes=ksizes,
        residual=False).to(device)

    if LOAD:
        net.load_state_dict(torch.load(loaded_model_path))

    if OPTIMIZER=='ADAM':
        optimizer = optim.Adam(net.parameters(), lr=LR)
    elif OPTIMIZER=='SGD':
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    if LOSS=='MSE':
        criterion = nn.MSELoss()
        losses_param = [nn.MSELoss()]
    elif LOSS=='L1':
        criterion = nn.L1Loss()
        losses_param = [nn.L1Loss()]
    elif LOSS=='NMSE':
        criterion = nmse_torch
        losses_param = [nmse_torch]
    if CYC_ONLY is True:
        losses_param = []

    losses_meas = [cycle_loss] if CYC_LOSS else []
    lam = [1,cyc_lam] if CYC_LOSS else [1]
    lam = [1] if CYC_ONLY else lam

    criterion = combine_losses(
        losses_param=losses_param,
        losses_meas=losses_meas,
        lam=lam, 
        outch_type=net.outch_type
    )

    training_summary = [
    '############## Network Parameters ############## \n',
    f'Number of starting filters = {NUM_FILT} \n',
    'Bilinear Interpolation for Upsampling (if False, use transposed ',
    f'convolution) = {BILINEAR} \n',
    f'Kernel Size = {ksizes} \n',
    f'Number of Layers = {numlayers} \n',
    f'Number of Detectors = {numdetectors} \n',
    f'Noise Model / dbsnr = {noise_model} / {dbsnr} \n',
    f'Output Channels = {OUTCH} \n',
    '\n############## Optimization Parameters ############## \n',
    f'Optimizer = {OPTIMIZER} \n',
    f'Loss = {LOSS} \n',
    f'Cycle Loss = {CYC_LOSS} \n',
    f'Cycle Loss Lam = {cyc_lam} \n',
    f'Learning Rate = {LR} \n',
    f'Num Epochs = {EPOCHS} \n',
    f'Training Batch Size = {BATCH_SIZE} \n',
    '\n############## Data Parameters ############## \n',
    f'Num of Tranining Images = {len(trainset)*5 if otf else len(trainset)} \n',
    f'Num of Validation Images = {len(valset)} \n',
    f'Dataset Path = {dataset_path} \n',
    ]

    with open(f'../results/saved/{name}/summary.txt', 'w') as file:
        for line in training_summary:
            file.write(line)

    try:
        t0 = time.time()
        (trainloss, valloss, train_ssim_eps, val_ssim_eps, train_rmse_eps, 
        val_rmse_eps, net) = train_net(net=net,
                  device=device,
                  trainloader=trainloader,
                  otf=otf,
                  valloader=valloader,
                  epochs=EPOCHS,
                  optimizer=optimizer,
                  criterion=criterion,
                  path=name
                  )
        train_time = datetime.timedelta(seconds=int(time.time() - t0))

    except KeyboardInterrupt:
        torch.save(net.state_dict(), f'../results/saved/{name}/INTERRUPTED.pth')
        sys.exit(0)

    torch.save(net.state_dict(), f'../results/saved/{name}/nf_{NUM_FILT}_LR_{LR}_EP_{EPOCHS}.pth')

    plt.figure()
    plt.semilogy(trainloss, label='Training Loss')
    plt.semilogy(valloss, label='Validation Loss')
    plt.title('Convergence Plot')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.savefig(f'../results/saved/{name}/convergence_plot.png')
    plt.close()

    plt.figure()
    plt.title('SSIMs vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('SSIM')
    plt.plot(train_ssim_eps[:,0], marker='$t$', color='navy', label='Int train')
    plt.plot(val_ssim_eps[:,0], marker='$v$', color='darkturquoise')
    plt.plot(train_ssim_eps[:,1], marker='$t$', color='darkred', label='Vel train')
    plt.plot(val_ssim_eps[:,1], marker='$v$', color='tomato')
    plt.plot(train_ssim_eps[:,2], marker='$t$', color='darkgreen', label='Width train')
    plt.plot(val_ssim_eps[:,2], marker='$v$', color='lime')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.savefig(f'../results/saved/{name}/ssims_vs_epochs.png')
    plt.close()

    plt.figure()
    plt.title('RMSEs vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('RMSE')
    plt.plot(train_rmse_eps[:,0], marker='$t$', color='navy', label='Int train')
    plt.plot(val_rmse_eps[:,0], marker='$v$', color='darkturquoise')
    plt.plot(train_rmse_eps[:,1], marker='$t$', color='darkred', label='Vel train')
    plt.plot(val_rmse_eps[:,1], marker='$v$', color='tomato')
    plt.plot(train_rmse_eps[:,2], marker='$t$', color='darkgreen', label='Width train')
    plt.plot(val_rmse_eps[:,2], marker='$v$', color='lime')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.savefig(f'../results/saved/{name}/rmses_vs_epochs.png')
    plt.close()

    net.load_state_dict(torch.load(f'../results/saved/{name}/best_model.pth'))
    net.eval()

    savedir = f'../results/saved/{name}/'
    ssims, rmses, yvec, outvec = plot_val_stats(net, testloader, savedir)
    plot_recons(net, testloader, numim=32, savedir=savedir+'figures/', denormalize=True)
    dbsnr_l = [10,20,30,None]
    ssims_l, rmses_l = eval_snrlist(dbsnr_list=dbsnr_l, noise_model=noise_model, fold='test', 
    data_dir=testset_path, net=net)
    labels_gr = ['int','vel','width'] if net.outch_type=='all' else [net.outch_type]
    barplot_group(ssims_l.mean(axis=1).swapaxes(0,1), 
        labels_gr=labels_gr, labels_mem=[str(jj) for jj in dbsnr_l], 
        ylabel='SSIM', title='SSIM vs dBsnr', savedir=savedir+'snr_barplot.png')
    est_bias = np.mean(outvec - yvec, axis=1) 
    est_std = np.std(outvec - yvec, axis=1)
    np.save(savedir+'ssims_l.npy', ssims_l)
    np.save(savedir+'rmses_l.npy', rmses_l)

    ######### Flower Dataset Evals ##########
    # testset = BasicDataset(data_dir=dataset_path, fold='test', dbsnr=dbsnr)
    # testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)

    # os.mkdir(f'../results/saved/{name}/test_results_flower')
    # savedir = f'../results/saved/{name}/test_results_flower/'
    # _ = plot_val_stats(net, testloader, savedir)
    # plot_recons(net, testloader, numim=32, savedir=savedir+'figures/')
    # ssims_lt, rmses_lt = eval_snrlist(dbsnr_list=dbsnr_l, fold='test', 
    # data_dir=dataset_path, net=net)
    # barplot_group(ssims_lt.mean(axis=1).swapaxes(0,1), 
    #     labels_gr=['int','vel','width'], labels_mem=[str(jj) for jj in dbsnr_l], 
    #     ylabel='SSIM', title='SSIM vs dBsnr', savedir=savedir+'snr_barplot.png')
    # np.save(savedir+'ssims_l.npy', ssims_lt)
    # np.save(savedir+'rmses_l.npy', rmses_lt)

    # os.mkdir(f'../results/saved/{name}/train_results')
    # savedir = f'../results/saved/{name}/train_results/'
    # _ = plot_val_stats(net, trainloader, savedir)
    # plot_recons(net, trainloader, numim=32, savedir=savedir+'figures/')
    # ssims_l, rmses_l = eval_snrlist(dbsnr_list=dbsnr_l, fold='val', 
    # data_dir=dataset_path, net=net)
    # barplot_group(ssims_l.mean(axis=1).swapaxes(0,1), 
    #     labels_gr=['int','vel','width'], labels_mem=[str(jj) for jj in dbsnr_l], 
    #     ylabel='SSIM', title='SSIM vs dBsnr', savedir=savedir+'snr_barplot.png')
    # np.save(savedir+'ssims_l.npy', ssims_l)
    # np.save(savedir+'rmses_l.npy', rmses_l)
    ##########################################

    training_summary += [
    '\n############## Train-Oracle Constant Estimates ############## \n',
    'SSIM Train: i:{:.3f}   v:{:.3f}    w:{:.3f} \n'.format(ssim_cnst_train[0], 
        ssim_cnst_train[1], ssim_cnst_train[2]),
    'SSIM Val: i:{:.3f}   v:{:.3f}    w:{:.3f} \n'.format(ssim_cnst_val[0], 
        ssim_cnst_val[1], ssim_cnst_val[2]),
    'SSIM Test: i:{:.3f}   v:{:.3f}    w:{:.3f} \n'.format(ssim_cnst_test[0], 
        ssim_cnst_test[1], ssim_cnst_test[2]),
    'RMSE Train: i:{:.4f}   v:{:.4f}    w:{:.4f} \n'.format(rmse_cnst_train[0], 
        rmse_cnst_train[1], rmse_cnst_train[2]),
    'RMSE Val: i:{:.4f}   v:{:.4f}    w:{:.4f} \n'.format(rmse_cnst_val[0], 
        rmse_cnst_val[1], rmse_cnst_val[2]),
    'RMSE Test: i:{:.4f}   v:{:.4f}    w:{:.4f} \n'.format(rmse_cnst_test[0], 
        rmse_cnst_test[1], rmse_cnst_test[2]),
    'MAE Train: i:{:.3f}   v:{:.3f}    w:{:.3f} \n'.format(mae_cnst_train[0], 
        mae_cnst_train[1], mae_cnst_train[2]),
    'MAE Val: i:{:.3f}   v:{:.3f}    w:{:.3f} \n'.format(mae_cnst_val[0], 
        mae_cnst_val[1], mae_cnst_val[2]),
    'MAE Test: i:{:.3f}   v:{:.3f}    w:{:.3f} \n'.format(mae_cnst_test[0], 
        mae_cnst_test[1], mae_cnst_test[2]),
    ]

    training_summary += [
    '\n############## Results ############## \n',
    'Final Validation Loss: {:.7f} \n'.format(valloss[-1]),
    'Minimum Validation Loss: {:.7f} \n'.format(np.min(valloss)),
    'Final Training Loss: {:.7f} \n'.format(trainloss[-1]),
    'Minimum Training Loss: {:.7f} \n\n'.format(np.min(trainloss)),
    ]

    with open(f'../results/saved/{name}/summary.txt', 'w') as file:
        for line in training_summary:
            file.write(line)

    if net.outch_type == 'all':
        training_summary += [
            'SSIM Mean+/-Std: i: {:.3f}+/-{:.3f}   v: {:.3f}+/-{:.3f}   w: {:.3f}+/-{:.3f} \n'.format(
                ssims[:,0].mean(), ssims[:,0].std(), ssims[:,1].mean(), ssims[:,1].std(), 
                ssims[:,2].mean(), ssims[:,2].std()),
            'RMSE Mean+/-Std: i: {:.4f}+/-{:.4f}   v: {:.4f}+/-{:.4f}   w: {:.4f}+/-{:.4f} \n'.format(
                rmses[:,0].mean(), rmses[:,0].std(), rmses[:,1].mean(), rmses[:,1].std(), 
                rmses[:,2].mean(), rmses[:,2].std()),
            'Bias+/-Std: i: {:.4f}+/-{:.4f}   v: {:.4f}+/-{:.4f}   w: {:.4f}+/-{:.4f} \n'.format(
                est_bias[0], est_std[0], rmses[:,1].mean(), rmses[:,1].std(), 
                rmses[:,2].mean(), rmses[:,2].std())
        ]
    elif net.outch_type == 'int':
        training_summary += [
            'Intensity SSIM Mean: {:.3f} \n'.format((ssims.mean())),
            'Intensity SSIM Std: {:.3f} \n'.format((ssims.std())),
            'Intensity RMSE Mean: {:.4f} \n'.format((rmses.mean())),
            'Intensity RMSE Std: {:.4f} \n'.format((rmses.std())),
            'Intensity Bias: {:.4f} \n'.format((est_bias[0])),
            'Intensity Error Std: {:.4f} \n\n'.format((est_std[0])),
        ]
    elif net.outch_type == 'vel':
        training_summary += [
            'Velocity SSIM Mean: {:.3f} \n'.format((ssims.mean())),
            'Velocity SSIM Std: {:.3f} \n'.format((ssims.std())),
            'Velocity RMSE Mean: {:.4f} \n'.format((rmses.mean())),
            'Velocity RMSE Std: {:.4f} \n'.format((rmses.std())),
            'Velocity Bias: {:.4f} \n'.format((est_bias[0])),
            'Velocity Error Std: {:.4f} \n\n'.format((est_std[0])),
        ]
    elif net.outch_type == 'width':
        training_summary += [
            'Linewidth SSIM Mean: {:.3f} \n'.format((ssims.mean())),
            'Linewidth SSIM Std: {:.3f} \n'.format((ssims.std())),
            'Linewidth RMSE Mean: {:.4f} \n'.format((rmses.mean())),
            'Linewidth RMSE Std: {:.4f} \n'.format((rmses.std())),
            'Linewidth Bias: {:.4f} \n'.format((est_bias[0])),
            'Linewidth Error Std: {:.4f} \n\n'.format((est_std[0])),
        ]

    training_summary += [
        f'Training Time: {str(train_time)} \n',
        '\n############## Notes ############## \n',
        ' \n',
        '\n############## Comments ############## \n'
    ]

    with open(f'../results/saved/{name}/summary.txt', 'w') as file:
        for line in training_summary:
            file.write(line)