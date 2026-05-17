import numpy as np
from slitless.forward import Imager, datacube_generator
from slitless.recon import smart2
from slitless.config import config

path_data = str(config.data_root / 'datasets/baseline')
data = np.load(path_data + 'eis_train_50_dsetv5.npy', allow_pickle=True).item()

DISP = 0.022275; C = 299792.458; RW = 195.117937907451; L = 21
WID = 42.74*(195.11794/C)
WID_PIX = WID / DISP
FRAC1 = 0.8555; FRAC2 = 0.0521; FRAC_BG = 0.0924
CENT2 = 195.17803
BG_SHAPE = np.ones(L) / L

Imgr = Imager(pixelated=True, spectral_orders=[0, -1, 1], intenscale=1)

def cent1_from_vel(v_km_s):
    return RW * (1 + v_km_s / C)

results = {}
for cent1_vel in [-1, -4]:
    label = f'cent1={cent1_vel:+d}'
    vel_all = []; wid_all = []; ve_edges = []; vc_center = []; we_edges = []; wc_center = []

    cent1 = cent1_from_vel(cent1_vel)

    for idx in range(len(data['param3d'])):
        truth = data['param3d'][idx]; meas = data['meas_damped'][idx]
        Imgr.meas3dar_nn = meas[:3]; Imgr.meas3dar = meas[:3]

        recon1, _ = smart2(imager=Imgr, fitter='mpfit', psi=0.2,
            maxouter=5, maxinner=20, prior_weight=0,
            cent1=cent1, wid1=WID, wid2=WID, n_jobs=-1)

        int0 = meas[0]
        vel2_off = (CENT2 - cent1) / DISP
        v1_0 = recon1[1]; w1_0 = WID_PIX * np.ones_like(int0)
        c1 = datacube_generator(np.stack((int0*FRAC1, v1_0, w1_0), axis=0), lamdim=L)
        v2_0 = v1_0 + vel2_off; w2_0 = WID_PIX * np.ones_like(int0)
        c2 = datacube_generator(np.stack((int0*FRAC2, v2_0, w2_0), axis=0), lamdim=L)
        bg = BG_SHAPE[:,np.newaxis,np.newaxis] * (int0*FRAC_BG)[np.newaxis,:,:]
        init_cube = c1 + c2 + bg

        recon2, _ = smart2(imager=Imgr, fitter='mpfit', psi=0.2,
            maxouter=5, maxinner=20, prior_weight=1, init_cube=init_cube,
            cent1=cent1, wid1=WID, wid2=WID, n_jobs=-1)

        r2 = Imgr.frompix(recon2, width_unit='km/s', array=True)
        ve = r2[1] - truth[1]
        we = r2[2] - truth[2] * C / RW
        vel_all.append([np.sqrt(np.mean(ve**2)), np.mean(ve)])
        wid_all.append([np.sqrt(np.mean(we**2)), np.mean(we)])

        for region, ys in [('edge', [slice(0,10), slice(54,64)]), ('center', [slice(27,37)])]:
            for sl in ys:
                if region == 'edge':
                    ve_edges.append(ve[sl,:].flatten())
                    we_edges.append(we[sl,:].flatten())
                else:
                    vc_center.append(ve[sl,:].flatten())
                    wc_center.append(we[sl,:].flatten())

    v = np.array(vel_all); w = np.array(wid_all)
    results[label] = {
        'vel_rmse': v[:,0].mean(), 'vel_bias': v[:,1].mean(),
        'wid_rmse': w[:,0].mean(), 'wid_bias': w[:,1].mean(),
        've': np.concatenate(ve_edges), 'vc': np.concatenate(vc_center),
        'we': np.concatenate(we_edges), 'wc': np.concatenate(wc_center),
    }

print("\n" + "=" * 70)
print("  TWO-STAGE SMART2: cent1 comparison (50 samples)")
print("=" * 70)
print(f"{'':>20} {'Vel RMSE':>8} {'Vel Bias':>8} {'Wid RMSE':>8} {'Wid Bias':>8}")
for label in ['cent1=-1', 'cent1=-4']:
    r = results[label]
    print(f"  {label:>18}: {r['vel_rmse']:8.2f} {r['vel_bias']:8.2f} {r['wid_rmse']:8.2f} {r['wid_bias']:8.2f} km/s")

print(f"\n  {'':>18}  {'Edge Vel':>8} {'Edge Bias':>7} {'Ctr Vel':>8} {'Ctr Bias':>7}  {'Edge Wid':>8} {'Ctr Wid':>8}")
for label in ['cent1=-1', 'cent1=-4']:
    r = results[label]
    print(f"  {label:>18}: {np.sqrt(np.mean(r['ve']**2)):8.2f} {np.mean(r['ve']):7.2f} {np.sqrt(np.mean(r['vc']**2)):8.2f} {np.mean(r['vc']):7.2f}  {np.sqrt(np.mean(r['we']**2)):8.2f} {np.sqrt(np.mean(r['wc']**2)):8.2f} km/s")
