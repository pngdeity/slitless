import torch, copy, glob, time, datetime, pickle, os, eispac
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from slitless.config import config
from slitless.forward import (Source, Imager, forward_op_torch, forward_op, 
    forward_op_tomo_3d, forward_op_tomo_3d_transpose, add_noise, gauss_pix,
    datacube_generator, tomomtx_gen)
from slitless.measure import cycle_loss, compare_ssim, tv_loss
from slitless.evaluate import net_loader, predict
from slitless.data_loader import meas_transform as unet_meas_transform, param_inv_transform
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import convolve
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class Reconstructor():
    def __init__(
        self,
        *,
        imager=None,
        solver=None,
        intenscale=1,
        simulate_meas=True,
        **solver_params
    ):
        self.imager = copy.deepcopy(imager)
        self.source = self.imager.srpix 
        self.solver = solver
        self.intenscale = intenscale
        self.solver_params = solver_params
        self.simulate_meas = simulate_meas
        self.tomo = True if hasattr(self.solver, '__name__') and self.solver.__name__ in ['smart', 'smart2', 'tomoinv'] else False

    def solve(
        self,
        num_realizations=1
    ):
        self.num_realizations = num_realizations
        if self.simulate_meas:
            _ = self.imager.get_measurements(
                dbsnr=self.imager.dbsnr, max_count=self.imager.max_count, 
                avg_count=self.imager.avg_count,
                noise_model=self.imager.noise_model, tomo=self.tomo
            )
        recons = []
        losses = []
        times = []
        for i in range(num_realizations):
            self.imager.meas3dar = add_noise(
                self.imager.meas3dar_nn, dbsnr=self.imager.dbsnr, 
                max_count=self.imager.max_count, avg_count=self.imager.avg_count, 
                noise_model=self.imager.noise_model
            )

            if self.simulate_meas is False:
                self.meas_transform()
            t0 = time.time()
            recon, loss = self.solver(
                imager = self.imager,
                **self.solver_params
            )
            t1 = time.time()
            if self.simulate_meas is False:
                recon = self.recon_inv_transform(recon)

            times.append(t1-t0)
            recons.append(recon)
            losses.append(loss)
        self.recons = Recon(recon=np.array(recons), losses=np.array(losses), 
            times=np.array(times), imager=self.imager, source=self.source,
            intenscale=self.intenscale)
        self.recons.eval()
        self.times = np.array(times)

        return self.recons
    
    def meas_transform(self):
        if self.solver.__name__=='nn_solver':
            self.imager.meas3dar = unet_meas_transform(
                self.imager.meas3dar.copy()*self.intenscale
            )
        
    def recon_inv_transform(self, recon):
        if self.solver.__name__=='nn_solver':
            return self.imager.topix(
                Source(
                    param3d=param_inv_transform(recon, w_kms=False),
                    pix=False
                )
            ).param3d
        else:
            return recon

class Reconstructor_Multi():
    def __init__(
        self,
        *,
        meas4dar=None,
        imager=None,
        param4dar=None,
        solver=None,
        pix=None,
        intenscaling=False,
        **solver_params
    ):
        self.imager = imager 
        self.meas4dar = meas4dar
        self.solver = solver
        self.param4dar = param4dar
        self.solver_params = solver_params
        self.intenscaling = intenscaling
        self.pix = pix
        self.simulate_meas = False if self.meas4dar is not None else True

    def solve(
        self,
        num_realizations=1
    ):
        self.num_realizations = num_realizations

        self.recons = []
        self.sources = []
        self.times = []

        self.ssim = []
        self.rmse_pix = []
        self.rmse_phy = []
        self.mae_pix = []
        self.mae_phy = []
        self.bias_pix = []
        self.bias_phy = []
        self.ssim_m = []
        self.rmse_pix_m = []
        self.rmse_phy_m = []
        self.mae_pix_m = []
        self.mae_phy_m = []

        for i in range(self.param4dar.shape[0]):
            Sr = Source(
                param3d=self.param4dar[i],
                pix=self.pix
            )

            self.sources.append(Sr)

            intenscale = Sr.param3d[0].max() if self.intenscaling else 1
            self.imager.intenscale=intenscale
            
            if self.pix==False:
                self.imager.topix(Sr)
            else:
                self.imager.srpix = Sr

            if self.meas4dar is not None:
                self.imager.meas3dar_nn = self.meas4dar[i] / intenscale
                self.imager.meas3dar = self.meas4dar[i] / intenscale
            
            Rec = Reconstructor(
                imager=self.imager,
                solver=self.solver,
                intenscale=intenscale,
                simulate_meas=self.simulate_meas,
                **self.solver_params
            )

            recons = Rec.solve(num_realizations=self.num_realizations)
            self.recons.append(recons)
            self.times.append(Rec.times)

            self.ssim.append(recons.ssim)
            self.rmse_pix.append(recons.rmse_pix)
            self.rmse_phy.append(recons.rmse_phy)
            self.mae_pix.append(recons.mae_pix)
            self.mae_phy.append(recons.mae_phy)
            self.bias_pix.append(recons.bias_pix)
            self.bias_phy.append(recons.bias_phy)
            self.ssim_m.append(recons.ssim_m)
            self.rmse_pix_m.append(recons.rmse_pix_m)
            self.rmse_phy_m.append(recons.rmse_phy_m)
            self.mae_pix_m.append(recons.mae_pix_m)
            self.mae_phy_m.append(recons.mae_phy_m)

        self.times = np.array(self.times)
        self.ssim = np.array(self.ssim)
        self.rmse_pix = np.array(self.rmse_pix)
        self.rmse_phy = np.array(self.rmse_phy)
        self.mae_pix = np.array(self.mae_pix)
        self.mae_phy = np.array(self.mae_phy)
        self.bias_pix = np.array(self.bias_pix)
        self.bias_phy = np.array(self.bias_phy)
        self.ssim_m = np.array(self.ssim_m)
        self.rmse_pix_m = np.array(self.rmse_pix_m)
        self.rmse_phy_m = np.array(self.rmse_phy_m)
        self.mae_pix_m = np.array(self.mae_pix_m)
        self.mae_phy_m = np.array(self.mae_phy_m)
        
        return self.recons


class Recon():
    def __init__(
            self,
            *,
            recon=None,
            losses=None,
            times=None,
            imager=None,
            source=None,
            intenscale=1
    ):
        self.recon = recon
        self.losses = losses
        self.times = times
        self.imager = imager
        self.source = source
        self.intenscale = intenscale
        self.losses_avg = np.mean(self.losses, axis=0)
        self.times_avg = np.mean(self.times)

    def plot(
            self,
            compare=False,
            index=0,
            title=''
    ):
        sr = Source(
            param3d = self.recon[index],
            pix=True
        )
        if compare is True:
            assert self.source is not None, "Source is not given!"
            truth = self.source.param3d

            fig, ax = plt.subplots(2,3, figsize=(15,8))
            plt.suptitle(title)
            i0=ax[0,0].imshow(sr.inten, cmap='hot')
            ax[0,0].set_title('Intensity')
            fig.colorbar(i0, ax=ax[0,0])
            i0=ax[0,1].imshow(sr.vel, cmap='seismic')
            fig.colorbar(i0, ax=ax[0,1])
            ax[0,1].set_title('Velocity [pix]')
            i0=ax[0,2].imshow(sr.width, cmap='plasma')
            fig.colorbar(i0, ax=ax[0,2])
            ax[0,2].set_title('Linewidth [pix]')

            i0=ax[1,0].imshow(truth[0], cmap='hot')
            ax[1,0].set_title('True Intensity')
            fig.colorbar(i0, ax=ax[1,0])
            i0=ax[1,1].imshow(truth[1], cmap='seismic')
            fig.colorbar(i0, ax=ax[1,1])
            ax[1,1].set_title('True Velocity [pix]')
            i0=ax[1,2].imshow(truth[2], cmap='plasma')
            fig.colorbar(i0, ax=ax[1,2])
            ax[1,2].set_title('True Linewidth [pix]')

            plt.tight_layout()
            plt.show()
        else:
            fig, ax = sr.plot(title=title)
        
        return fig, ax


    def plot_loss(self):
        plt.figure()
        plt.title('Loss vs Iter')
        plt.plot(self.losses_avg, linewidth=2)
        plt.grid(which='both', axis='both')
        plt.show()

    def eval(self):
        truth_pix = self.source.param3d
        truth_pix_mean = np.ones_like(truth_pix) * truth_pix.mean(axis=(1,2))[:,None,None]
        recon_pix = self.recon
        truth_pix = np.repeat(truth_pix[np.newaxis,:], len(recon_pix), axis=0)
        truth_pix_mean = np.repeat(truth_pix_mean[np.newaxis,:], len(recon_pix), axis=0)
        truth_phy = self.imager.frompix(truth_pix, width_unit='km/s', array=True)
        truth_phy_mean = self.imager.frompix(truth_pix_mean, width_unit='km/s', array=True)
        recon_phy = self.imager.frompix(recon_pix, width_unit='km/s', array=True)  

        self.ssim = compare_ssim(truth=truth_pix, estimate=recon_pix)
        self.rmse_pix = np.sqrt(np.mean((recon_pix-truth_pix)**2, axis=(-1,-2)))
        self.rmse_phy = np.sqrt(np.mean((recon_phy-truth_phy)**2, axis=(-1,-2)))
        self.mae_pix = np.mean(abs(recon_pix-truth_pix), axis=(-1,-2))
        self.mae_phy = np.mean(abs(recon_phy-truth_phy), axis=(-1,-2))
        self.bias_pix = np.mean(recon_pix-truth_pix, axis=(-1,-2))
        self.bias_phy = np.mean(recon_phy-truth_phy, axis=(-1,-2))
        self.ssim_m = compare_ssim(truth=truth_pix, estimate=truth_pix_mean)
        self.rmse_pix_m = np.sqrt(np.mean((truth_pix_mean-truth_pix)**2, axis=(-1,-2)))
        self.rmse_phy_m = np.sqrt(np.mean((truth_phy_mean-truth_phy)**2, axis=(-1,-2)))
        self.mae_pix_m = np.mean(abs(truth_pix_mean-truth_pix), axis=(-1,-2))
        self.mae_phy_m = np.mean(abs(truth_phy_mean-truth_phy), axis=(-1,-2))

def gauss_pmf_fitter(
    line
):
    inten = np.sum(line, axis=0)
    line0 = line / inten
    mean = np.sum(np.arange(len(line))[:,None,None]*line0, axis=0)
    std = np.sqrt(np.sum((np.arange(len(line))**2)[:,None,None]*line0, axis=0) - mean**2)
    mean -= len(line)//2
    return np.stack((inten, mean, std), axis=0)

def gauss_pmf_fitter2(
    line
):
    inten = np.sum(line, axis=0)
    mean = np.zeros_like(inten)
    std = np.ones_like(inten)
    ind0 = np.where(inten<=0)

    mean = np.sum(np.arange(len(line))[:,None,None]*line, axis=0) / inten
    std = np.sqrt(np.sum((np.arange(len(line))**2)[:,None,None]*line, axis=0) / inten - mean**2)
    mean -= len(line)//2
    inten[ind0] = 0
    std[ind0] = 1.2
    mean[ind0] = 0
    inten = np.clip(inten, 0, 1)
    std = np.clip(std, 0.5, 2.3)
    mean = np.clip(mean, -2, 2)
    inten[np.isnan(inten)] = 0
    mean[np.isnan(mean)] = 0
    std[np.isnan(std)] = 2.3

    return np.stack((inten, mean, std), axis=0)

def gauss_(x,inten, vel, width):
    return inten * gauss_pix(x,vel+len(x)//2,width)

def gauss_curvefit(dc):
    M, N, R = dc.shape
    param3d = np.zeros((3,N,R))
    for i in range(N):
        for j in range(R):
            par, var = curve_fit(gauss_, np.arange(M), dc[:,i,j], 
                p0=[1,0,1], bounds=((0,-2,1),(1,2,2.3)), maxfev=5000)
            param3d[:,i,j] = par
    return param3d

def smart_fit_spectra_joblib(cube, tmplt, wave, errs=None, n_jobs=-1, component=0):
    """
    Fits a reconstructed 3D data cube from SMART using mpfit and an eispac template.
    
    Args:
        cube: 3D numpy array of shape (lamdim, Y, X)
        tmplt: eispac template object
        wave: 1D array of shape (lamdim,) containing wavelength values
        errs: 3D array of uncertainties. If None, assumes uniform errors.
        n_jobs: Number of parallel jobs
        component: Gaussian component index to extract from the template
        
    Returns:
        param3d: Stacked array (3, Y, X) containing Intensity, Velocity (km/s), and Width (Angstroms).
    """
    import copy
    from joblib import Parallel, delayed
    from slitless.eistools import _worker_fit_chunk
    from eispac.instr import calc_velocity

    safe_data = np.transpose(cube, (1, 2, 0)).astype(np.float64)
    Y, X, lamdim = safe_data.shape
    
    if errs is None:
        safe_errs = np.ones_like(safe_data)
    else:
        safe_errs = np.transpose(errs, (1, 2, 0)).astype(np.float64)
        
    if wave.ndim == 1:
        safe_wave = np.tile(wave, (Y, X, 1)).astype(np.float64)
    else:
        safe_wave = np.transpose(wave, (1, 2, 0)).astype(np.float64)

    p_base = tmplt.parinfo
    t_base = tmplt.template
    
    tasks = []
    for y in range(Y):
        tasks.append((
            safe_wave[y, :, :], 
            safe_data[y, :, :], 
            safe_errs[y, :, :], 
            t_base, 
            copy.deepcopy(p_base), 
            7 # min_points
        ))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_worker_fit_chunk)(t) for t in tasks
    )
    
    full_params = np.stack([r[0] for r in results])
    full_status = np.stack([r[2] for r in results])

    idx_peak = component * 3
    idx_cent = component * 3 + 1
    idx_width = component * 3 + 2

    raw_peak = full_params[:, :, idx_peak]
    raw_cent = full_params[:, :, idx_cent]
    raw_width = full_params[:, :, idx_width]

    intensity = np.sqrt(2 * np.pi) * raw_peak * raw_width
    rest_wave = p_base[idx_cent]['value'] 
    
    # Calculate absolute velocity to bypass eispac's orbital thermal drift correction
    velocity = config.speed_of_light * (raw_cent - rest_wave) / rest_wave

    bad_mask = (full_status <= 0)
    intensity[bad_mask] = 0
    velocity[bad_mask] = 0
    raw_width[bad_mask] = 0

    param3d = np.stack((intensity, velocity, raw_width))
    return param3d

def tomoinv0(
    meas=None,
    imager=None,
    stepsize=1e-1,
    lam=1e-1,
    numiter=20
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gauss_curvefit = gauss_pmf_fitter2
    # gauss_curvefit = lambda a : a
    # datacube_generator = lambda a : a * (a>=0).astype(a.dtype)
    if imager is not None:
        meas = torch.tensor(imager.meas3dar.copy()).to(device)
    else:
        meas = torch.tensor(meas).to(device)
    NK, N, R = meas.shape
    # H = torch.tensor(tomomtx_gen((21,N), orders=imager.spectral_orders), dtype=torch.float64).to(device)
    H = torch.tensor(tomomtx_gen((21,N), orders=[0,-1,1]), dtype=torch.float64).to(device)
    y = meas.view(-1,R)
    gaminv = torch.linalg.inv(H.T@H+lam*torch.eye(21*N).to(device))
    Hty = H.T @ y
    r_pinv = gaminv @ Hty
    r_hat = torch.tensor(datacube_generator(gauss_curvefit(r_pinv.view(21,N,R).cpu().numpy()))).to(device).view(-1,R)
    for i in tqdm(range(numiter)):
        r_hat = torch.tensor(datacube_generator(gauss_curvefit((r_pinv + lam * gaminv @ r_hat).view(21,N,R).cpu().numpy()))).to(device).view(-1,R)
    gauss_curvefit = gauss_pmf_fitter2
    return gauss_curvefit(r_hat.view(21,N,R).cpu().numpy()), []

def tomoinv(
    meas=None,
    imager=None,
    data_step='grad',
    positivity=False,
    proj='gauss',
    init_recon=None,
    stepsize=1e-1,
    lam=1e-1,
    numiter=20
):
    gauss_curvefit = gauss_pmf_fitter2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if imager is not None:
        meas = torch.tensor(imager.meas3dar.copy()).to(device)
    else:
        meas = torch.tensor(meas).to(device)
    NK, N, R = meas.shape
    H = torch.tensor(tomomtx_gen((21,N), orders=[0,-1,1]), dtype=torch.float64).to(device)
    y = meas.view(-1,R)
    Hty = H.T @ y
    loss = []

    if proj == 'gauss':
        proj0 = lambda a : torch.tensor(datacube_generator(gauss_curvefit(a.view(21,N,R).cpu().numpy()))).to(device).view(-1,R)
        if positivity:
            def proj(a):
                return proj0((a*(a>0)).to(a.dtype))
        else:
            proj = proj0
    elif proj == 'positivity':
        proj = lambda a: (a*(a>0)).to(a.dtype)
    if data_step == 'inv':
        gaminv = torch.linalg.inv(H.T@H+lam*torch.eye(21*N).to(device))
        r_pinv = gaminv @ Hty
        r_hat = proj(r_pinv)
        for i in tqdm(range(numiter)):
            r_hat = proj(r_pinv + lam * gaminv @ r_hat)
            lam *= 0.95
            loss.append(torch.norm(H@r_hat-y).cpu().numpy())
    elif data_step == 'grad':
        if init_recon is not None:
            r_hat = torch.tensor(init_recon, dtype=y.dtype).to(device).view(-1,R)
        else:
            r_hat = torch.zeros((21*N,R), dtype=torch.float64).to(device)
        for i in tqdm(range(numiter)):
            Hr = H@r_hat
            r_hat = proj(r_hat - stepsize*(H.T@Hr-Hty))
            loss.append(torch.norm(H@r_hat-y).cpu().numpy())
    gauss_curvefit = gauss_pmf_fitter2
    return gauss_curvefit(r_hat.view(21,N,R).cpu().numpy()), loss


def smart(
        meas=None,
        imager=None,
        psi=0.2,
        maxouter=5,
        maxinner=20,
        inf_prior_width=1.38,
        fitter='pmf',
        tmplt=None,
        n_jobs=-1
):
    if imager is not None:
        meas = imager.meas3dar.copy()
    NK, M, N = meas.shape
    if inf_prior_width is not None:
        infprior = gauss_pix(np.outer(np.arange(M),np.ones(M)), M//2, inf_prior_width)
        meas = np.concatenate((meas, infprior[None]), axis=0)
        meas[-1]*=meas[0].mean(axis=0)[None]/meas[-1].mean(axis=0)[None]
        NK += 1 
        
    orders = imager.spectral_orders
    inf=True if inf_prior_width is not None else False
    orders = orders+['inf'] if inf else orders
    #initialize
    cubes = []
    cors = []
    int0 = meas[0].copy()
    vel0 = np.zeros_like(int0)
    width0 = 1.38*np.ones_like(vel0)
    cube = datacube_generator(np.stack((int0,vel0,width0),axis=0))
    cubes.append(cube)
    k0 = np.array([0.25, 0.5, 0.25])
    k1 = np.outer(k0,k0)
    kernel = k0[None,None] * k1[:,:,None]

    mtx = tomomtx_gen((M,M), orders=orders)
    mtx_t = np.einsum('ijk->ikj', mtx.reshape(-1,M,M*M))
    mtx_s = (np.sum(mtx_t,axis=2)<1).astype(int).reshape(-1,M,M)[:,:,:,None]
    mtx_s = np.repeat(mtx_s, N, axis=3)

    for i in range(maxouter):
        print('Outer Iter: {}/{}'.format(i+1,maxouter))
        # contrast enhancement
        cube = (cube + cube**(1+psi))*np.sum(cube)/np.sum(cube + cube**(1+psi))
        # kernel smoothing
        cube = convolve(cube, kernel)
        for j in range(maxinner):
            # print('Inner Iter: {}/{}'.format(j+1,maxinner))
            # meas2 = forward_op_tomo_3d(cube, orders=orders, inf=inf)
            meas2 = (mtx @ cube.reshape(-1,N)).reshape(meas.shape)
            # chi-square
            chi = np.mean(((meas-meas2)**2)/(meas+1e-7), axis=(1,2))
            unconverged = chi>0.0000000001
            # if all are converged, go to the next iter
            if np.sum(unconverged)==0:
                continue
            # cor = (meas/(meas2+1e-5))**(2/(3+1*(inf==True)))
            cor = (meas/(meas2+1e-5))**(2/(3))
            # cor = (meas2/meas)**2/3 # Warning!: reversed order in ESIS2022
            # Cor = forward_op_tomo_3d_transpose(cor, orders=orders, inf=inf)
            Cor = np.einsum('ijk,ikm->ijm',mtx_t, cor).reshape(NK,M,M,N)
            Cor[mtx_s==1]=1

            Corr = np.prod(Cor[unconverged],axis=0)**(1/np.sum(unconverged))
            cube *= Corr
        print(f'chi:{chi}')
        # cors.append(Cor)
        # cubes.append(cube)
    
    if fitter=='mpfit':
        if tmplt is None:
            template_filepath = str(config.template_path)
            tmplt = eispac.read_template(template_filepath)
        lamdim = cube.shape[0]
        
        wave_cen = imager.mid_wavelength if imager is not None else config.mid_wavelength
        disp_scale = imager.dispersion_scale if imager is not None else config.dispersion_scale
        
        # Construct the wavelength grid assuming pixel units center around lamdim//2
        wave = wave_cen + disp_scale * (np.arange(lamdim) - lamdim // 2)
        cube = cube / disp_scale * imager.intenscale
        
        recon = smart_fit_spectra_joblib(cube, tmplt, wave=wave, n_jobs=n_jobs)
        
        # convert physical units (km/s, Angstroms) back to pixel units to match original format
        SPEED_OF_LIGHT = config.speed_of_light
        recon[0] /= imager.intenscale
        rest_wave = tmplt.parinfo[1]['value']
        actual_wave = rest_wave * (1 + recon[1] / SPEED_OF_LIGHT)
        recon[1] = (actual_wave - wave_cen) / disp_scale
        recon[2] = recon[2] / disp_scale
    elif fitter=='pmf':
        recon = gauss_pmf_fitter(cube)

    # recon[2] = 1.42
    # recon[1] = -0.01
    # recon[0] = int0
    
    return recon , cube

def prior_solver(
    meas=None,
    imager=None,
    frac1=0.8620,
    cent1=-0.95*(195.11723/299792.458)+195.11723,
    wid1=1.28*0.022275,
    **kwargs
):
    if imager is not None:
        meas = imager.meas3dar.copy()
        
    int0 = meas[0].copy()
    
    if imager is not None:
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale
        
    vel1_pix_0 = (cent1 - mid_wave) / disp_scale
    wid1_pix_0 = wid1 / disp_scale
    
    recon_int = int0 * frac1
    recon_vel = vel1_pix_0 * np.ones_like(int0)
    recon_wid = wid1_pix_0 * np.ones_like(int0)
    
    recon = np.stack((recon_int, recon_vel, recon_wid), axis=0)
    
    return recon, []

def smart2(
        meas=None,
        imager=None,
        psi=0.2,
        maxouter=5,
        maxinner=20,
        prior_weight=1.0,
        fitter='mpfit',
        tmplt=None,
        n_jobs=-1,
        frac1=0.8555,
        frac2=0.0521,
        frac_bg=0.0924,
        cent1=195.11803,
        wid1=0.02907,
        cent2=195.17803,
        wid2=0.02907,
        bg_shape_norm=[0.04762] * 21,
        init_cube=None,
):
    if imager is not None:
        meas = imager.meas3dar.copy()
    NK, M, N = meas.shape
    L = 21

    if tmplt is None and fitter == 'mpfit':
        template_filepath = str(config.template_path)
        import os
        if os.path.exists(template_filepath):
            import eispac
            tmplt = eispac.read_template(template_filepath)

    orders = imager.spectral_orders
    orders_list = list(orders)
    if prior_weight > 0:
        orders_list.append('inf')  # infinite projection (column-summed spectrum)

    meas_list = [meas[k] for k in range(NK)]
    
    int0 = meas[0].copy()
    
    if imager is not None:
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale
        
    vel1_pix_0 = (cent1 - mid_wave) / disp_scale
    wid1_pix_0 = wid1 / disp_scale
    
    vel2_pix_0 = (cent2 - mid_wave) / disp_scale
    wid2_pix_0 = wid2 / disp_scale

    if init_cube is not None:
        cube = init_cube.copy()  # skip 3-component initialization, use provided cube
    else:
        v1_0 = vel1_pix_0 * np.ones_like(int0)
        w1_0 = wid1_pix_0 * np.ones_like(int0)
        cube1 = datacube_generator(np.stack((int0 * frac1, v1_0, w1_0), axis=0), lamdim=L)

        v2_0 = vel2_pix_0 * np.ones_like(int0)
        w2_0 = wid2_pix_0 * np.ones_like(int0)
        cube2 = datacube_generator(np.stack((int0 * frac2, v2_0, w2_0), axis=0), lamdim=L)
        
        if bg_shape_norm is None:
            bg_shape_norm = np.ones(L) / L
        bg_cube = np.array(bg_shape_norm)[:, np.newaxis, np.newaxis] * (int0 * frac_bg)[np.newaxis, :, :]

        cube = cube1 + cube2 + bg_cube

    if prior_weight > 0:
        infprior = np.sum(cube, axis=1)  # column-summed spectrum (L x N)
        infprior = infprior / np.clip(infprior.sum(axis=0), 1e-12, None) * meas[0].sum(axis=0)
        meas_list.append(infprior)
    # when prior_weight=0, skip inf-prior entirely to avoid diverging chi

    num_projs = len(meas_list)
    
    weights = np.ones(num_projs)
    if prior_weight > 0:
        weights[-1] = prior_weight

    mtx_list = []
    for order in orders_list:
        mtx_list.append(tomomtx_gen((L, M), orders=[order]))

    mapped_list = []
    mtx_s_list = []
    for k in range(num_projs):
        mapped = mtx_list[k].T @ np.ones((mtx_list[k].shape[0], 1))
        mapped_list.append(mapped)
        mtx_s_list.append((mapped < 0.99).flatten())

    k0 = np.array([0.25, 0.5, 0.25])
    k1 = np.outer(k0,k0)
    kernel = k0[None,None] * k1[:,:,None]

    for i in range(maxouter):
        print('Outer Iter: {}/{}'.format(i+1,maxouter))
        cube = (cube + cube**(1+psi))*np.sum(cube)/np.sum(cube + cube**(1+psi))
        cube = convolve(cube, kernel, mode='reflect')
        for j in range(maxinner):
            cube_flat = cube.reshape(L*M, N)
            
            meas2_list = []
            for k in range(num_projs):
                meas2_list.append(mtx_list[k] @ cube_flat)
                
            chi_list = []
            for k in range(num_projs):
                chi_list.append(np.mean(((meas_list[k]-meas2_list[k])**2)/(meas_list[k]+1e-7)))
            chi = np.array(chi_list)
            unconverged = chi > 1e-10
            if np.sum(unconverged) == 0:
                continue
            
            Cor_list = []
            active_count = 0.0
                
            for k in range(num_projs):
                # cor_k = (meas_list[k]/(meas2_list[k]+1e-5))**(2/(num_projs))
                cor_k = meas_list[k]/(meas2_list[k]+1e-2)
                
                Cor_k_flat = (mtx_list[k].T @ cor_k) / (mapped_list[k] + 1e-2)
                Cor_k = Cor_k_flat.reshape(L, M, N)
                
                missing_mask = mtx_s_list[k].reshape(L, M, 1)
                
                # Core-Copy Boundary Condition: Ensure uniform spectral shape scaling
                fallback = Cor_k[L//2, :, :][None, :, :]
                Cor_k = np.where(missing_mask, fallback, Cor_k)
                
                # Apply the voting weight
                Cor_k = Cor_k ** weights[k]
                Cor_list.append(Cor_k)
                
                if unconverged[k]:
                    active_count += weights[k]
            
            Cor = np.stack(Cor_list, axis=0)
            Corr = np.prod(Cor[unconverged], axis=0) ** (1.0 / max(active_count, 1.0))
            cube *= Corr
                
        print(f'chi:{np.mean(chi)}')
        
    if fitter=='mpfit':
        wave_cen = imager.mid_wavelength if imager is not None else config.mid_wavelength
        disp_scale = imager.dispersion_scale if imager is not None else config.dispersion_scale
        wave = wave_cen + disp_scale * (np.arange(L) - L // 2)
        cube = cube / disp_scale * imager.intenscale
        
        recon = smart_fit_spectra_joblib(cube, tmplt, wave=wave, n_jobs=n_jobs)
        
        # convert physical units (km/s, Angstroms) back to pixel units to match original format
        SPEED_OF_LIGHT = config.speed_of_light
        recon[0] /= imager.intenscale
        rest_wave = tmplt.parinfo[1]['value']
        actual_wave = rest_wave * (1 + recon[1] / SPEED_OF_LIGHT)
        recon[1] = (actual_wave - wave_cen) / disp_scale
        recon[2] = recon[2] / disp_scale
    elif fitter=='pmf':
        bg = np.min(cube, axis=0, keepdims=True)
        cube_safe = np.clip(cube - bg, 0.0, None)
        recon = gauss_pmf_fitter2(cube_safe)

    return recon , cube * disp_scale

def smart2_twostage(
        imager=None,
        psi=0.2,
        maxouter=5,
        maxinner=20,
        prior_weight=1.0,
        fitter='mpfit',
        tmplt=None,
        n_jobs=-1,
        frac1=0.8555,
        frac2=0.0521,
        frac_bg=0.0924,
        cent1=195.11803,
        wid1=0.02907,
        cent2=195.17803,
        wid2=0.02907,
        bg_shape_norm=[0.04762] * 21,
        **kwargs,
):
    # Stage 1: pw=0 (no inf-prior) for better velocity, doubled iterations
    # Stage 2: pw=1 with init_cube built from stage-1 velocity for width constraint
    # Falls through to single-stage smart2 if prior_weight=0
    if prior_weight == 0:
        return smart2(
            imager=imager, psi=psi, maxouter=maxouter, maxinner=maxinner,
            prior_weight=0, fitter=fitter, tmplt=tmplt, n_jobs=n_jobs,
            frac1=frac1, frac2=frac2, frac_bg=frac_bg,
            cent1=cent1, wid1=wid1, cent2=cent2, wid2=wid2,
            bg_shape_norm=bg_shape_norm,
        )

    recon1, _ = smart2(
        imager=imager, psi=psi, maxouter=2*maxouter, maxinner=2*maxinner,
        prior_weight=0, fitter=fitter, tmplt=tmplt, n_jobs=n_jobs,
        frac1=frac1, frac2=frac2, frac_bg=frac_bg,
        cent1=cent1, wid1=wid1, cent2=cent2, wid2=wid2,
        bg_shape_norm=bg_shape_norm,
    )  # stage 1: pw=0, doubled iterations for better velocity

    # build 3-component init_cube using stage-1 velocity map
    if imager is not None:
        disp_scale = imager.dispersion_scale
        int0 = imager.meas3dar[0].copy()
    else:
        disp_scale = config.dispersion_scale
        int0 = np.zeros((64, 64))

    L = 21
    vel2_offset = (cent2 - cent1) / disp_scale
    wid_pix = wid1 / disp_scale

    v1_0 = recon1[1]
    w1_0 = wid_pix * np.ones_like(int0)
    cube1 = datacube_generator(np.stack((int0 * frac1, v1_0, w1_0), axis=0), lamdim=L)

    v2_0 = v1_0 + vel2_offset
    w2_0 = wid_pix * np.ones_like(int0)
    cube2 = datacube_generator(np.stack((int0 * frac2, v2_0, w2_0), axis=0), lamdim=L)

    if bg_shape_norm is None:
        bg_shape_norm = np.ones(L) / L
    bg_cube = np.array(bg_shape_norm)[:, np.newaxis, np.newaxis] * (int0 * frac_bg)[np.newaxis, :, :]
    init_cube = cube1 + cube2 + bg_cube

    recon2, cube = smart2(
        imager=imager, psi=psi, maxouter=maxouter, maxinner=maxinner,
        prior_weight=prior_weight, fitter=fitter, tmplt=tmplt, n_jobs=n_jobs,
        init_cube=init_cube,
        cent1=cent1, wid1=wid1, cent2=cent2, wid2=wid2,
        bg_shape_norm=bg_shape_norm,
    )  # stage 2: pw=1 with inf-prior, constrained by stage-1 velocity init
    return recon2, cube

def grad_descent_solver(
    imager=None,
    truth=None,
    OPTIMIZER = 'ADAM',
    USE_TV_LOSS = True,
    DATA_FIDELITY = 'L2', # 'L1' or 'L2'
    lam_i = 1e-2, # TV norm regularization parameter for intensity
    lam_v = 1e-2, # TV norm regularization parameter for velocity
    lam_w = 1e-2, # TV norm regularization parameter for width
    LR = 1e-2, # learning rate of the optimizer
    maxiter = 10000,
    # return_arrays=False, # if true, return every 1000th vel&width recon
    savepath=None
):
    """
    It solves for the intensity, velocity and width using gradient descent.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    meas = imager.meas3dar.copy()

    if type(meas) is not torch.Tensor:
        meas = torch.from_numpy(meas).to(device=device, dtype=torch.float)

    mask = imager.mask.copy()
    if type(mask) is not torch.Tensor:
        mask = torch.from_numpy(mask).to(device=device, dtype=torch.float)

    xh_int = copy.deepcopy(meas[...,0,:,:])
    xh_int = xh_int.requires_grad_()
    
    if imager is not None:
        rest_wave = imager.srpix.rest_wavelength if hasattr(imager, 'srpix') else config.wavelength
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        rest_wave = config.wavelength
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale
        
    vel_pix_0 = (rest_wave - mid_wave) / disp_scale
    width_pix_0 = 0.02888811 / disp_scale
    
    xh_vel = vel_pix_0 * torch.ones_like(
        xh_int, device=device, dtype=torch.float)
    xh_vel = xh_vel.requires_grad_()
    xh_width = width_pix_0 * torch.ones_like(xh_int, device=device, dtype=torch.float)
    xh_width = xh_width.requires_grad_()

    if OPTIMIZER.upper() == 'ADAM':
        optimizer = optim.Adam([xh_int, xh_vel, xh_width], lr=LR)
    if OPTIMIZER.upper() == 'SGD':
        optimizer = optim.SGD([xh_int, xh_vel, xh_width], lr=LR)
    xhs = []
    if truth is not None:
        diffs_vel = []
        diffs_width = []

    for i in tqdm(range(maxiter)):
        optimizer.zero_grad()
        if DATA_FIDELITY == 'L1':
            loss = torch.mean(abs(meas-imager.forward_op(xh_int,xh_vel,xh_width)))
        elif DATA_FIDELITY == 'L2': 
            loss = torch.mean((meas-imager.forward_op(xh_int,xh_vel,xh_width))**2)
        if USE_TV_LOSS:
            loss += lam_i*tv_loss(xh_int) + lam_v*tv_loss(xh_vel) + lam_w*tv_loss(xh_width)
        loss.backward()

        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        if truth is not None:
            diff_vel = torch.sum((truth[:,1]-xh_vel)**2)/torch.sum(truth[:,1]**2)
            diffs_vel.append(diff_vel.detach().cpu().numpy())

            diff_width = torch.sum((truth[:,2]-xh_width)**2)/torch.sum(truth[:,2]**2)
            diffs_width.append(diff_width.detach().cpu().numpy())

        if (savepath is not None) & (i>0) & (i%10000==0):
            xhs0 = torch.stack((xh_int,xh_vel,xh_width), axis=-3).detach().cpu().numpy()
            np.save(savepath+f'recons_{i}.npy', xhs0)

        # if return_arrays & (i>0) & (i%1000==0):
        #     xhs0 = torch.stack((xh_int,xh_vel,xh_width), axis=-3).detach().cpu().numpy()
        #     xhs.append(xhs0)

    # xhs = np.array(xhs)
    losses = np.array(losses)
    recon = torch.stack((xh_int,xh_vel,xh_width), axis=-3).detach().cpu().numpy()
    # recon = Recon(recon=recon, losses=losses, imager=imager)

    return recon, losses

def nn_solver(
        imager=None,
        model_path='2023_01_19__17_18_44_NF_64_BS_4_LR_0.0002_EP_200_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_35_dssize_full'
):
    foldpath = glob.glob(str(config.model_dir)+'/'+'*'+model_path+'*')[0]+'/'
    net = net_loader(foldpath)
    net.eval()
    recon = predict(net, imager.meas3dar.copy())

    losses = []
    return recon, losses

def scipy_solver(
    imager=None,
    OPTIMIZER = 'L-BFGS-B',
    DATA_FIDELITY = 'L2', # 'L1' or 'L2'
    lam_i = 5e2, # TV norm regularization parameter for intensity
    lam_v = 5e2, # TV norm regularization parameter for velocity
    lam_w = 1e0, # TV norm regularization parameter for width
    maxiter=10000
    ):

    def obj_ls(x, meas=None, mask=None, lam_i=1e1, lam_v=1e1, lam_w=1e1):
        aa, bb = meas.shape[1:]
        intensity, doppler, linewidth = np.reshape(x, (3,aa,bb))
        diff = forward_op(intensity, doppler, linewidth,
            pixelated=imager.pixelated, mask=mask,
            spectral_orders=imager.spectral_orders) - meas
        regu = (
            lam_v*np.sum(np.diff(doppler, axis=0)**2) + 
            lam_w*np.sum(np.diff(linewidth, axis=0)**2) +
            lam_i*np.sum(np.diff(intensity, axis=0)**2)
        )

        if DATA_FIDELITY == 'L2':
            return np.sum(diff**2) + regu
        elif DATA_FIDELITY == 'L1':
            return np.sum(abs(diff)) + regu

    meas = imager.meas3dar.copy()
    mask = imager.mask.copy()
    aa, bb = meas[0].shape

    int0 = meas[0].copy()
    
    if imager is not None:
        rest_wave = imager.srpix.rest_wavelength if hasattr(imager, 'srpix') else config.wavelength
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        rest_wave = config.wavelength
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale
        
    vel_pix_0 = (rest_wave - mid_wave) / disp_scale
    width_pix_0 = 0.02888811 / disp_scale
    
    vel0 = vel_pix_0 * np.ones_like(int0)
    width0 = width_pix_0 * np.ones_like(int0)
    x0 = np.stack((int0, vel0, width0), axis=0).flatten()

    rec = np.zeros((3,aa,bb))
    for i in tqdm(range(bb)):
        x0 = np.stack( ( int0[:,i], vel0[:,i], width0[:,i] ), axis=0 ).flatten()
        recon = minimize(
            obj_ls, 
            x0, 
            args=(meas[:,:,[i]], mask[:,[i]], lam_i, lam_v, lam_w), 
            method=OPTIMIZER,
            options={'disp':False, 'maxiter':maxiter}
        )
        rec[:,:,i] = recon.x.reshape(3,aa)

    losses = []
    return rec, losses

def _worker_scipy_col(x0, meas_slice, mask_slice, lam_i, lam_v, lam_w, 
                      pixelated, spectral_orders, OPTIMIZER, maxiter, DATA_FIDELITY):

    def obj_ls_local(x, meas, mask, lam_i, lam_v, lam_w):
        aa, bb = meas.shape[1:]
        
        intensity, doppler, linewidth = np.reshape(x, (3,aa,bb))
            
        proj_gauss = forward_op(intensity, doppler, linewidth,
            pixelated=pixelated, mask=mask,
            spectral_orders=spectral_orders)
        
        diff = proj_gauss - meas
        
        regu = (
            lam_v*np.sum(np.diff(doppler, axis=0)**2) + 
            lam_w*np.sum(np.diff(linewidth, axis=0)**2) +
            lam_i*np.sum(np.diff(intensity, axis=0)**2)
        )

        if DATA_FIDELITY == 'L2':
            return np.sum(diff**2) + regu
        elif DATA_FIDELITY == 'L1':
            return np.sum(abs(diff)) + regu

    res = minimize(
        obj_ls_local, 
        x0, 
        args=(meas_slice, mask_slice, lam_i, lam_v, lam_w), 
        method=OPTIMIZER,
        options={'disp':False, 'maxiter':maxiter}
    )
    return res.x

def scipy_solver_parallel(
    imager=None,
    OPTIMIZER = 'L-BFGS-B',
    DATA_FIDELITY = 'L2', # 'L1' or 'L2'
    lam_i = 5e2, # TV norm regularization parameter for intensity
    lam_v = 5e2, # TV norm regularization parameter for velocity
    lam_w = 1e0, # TV norm regularization parameter for width
    maxiter=10000,
    n_jobs=-1
    ):

    meas = imager.meas3dar.copy()
    mask = imager.mask
    if mask is None:
        mask = np.ones_like(meas[0])
    aa, bb = meas[0].shape

    int0 = meas[0].copy()

    if imager is not None:
        rest_wave = imager.srpix.rest_wavelength if hasattr(imager, 'srpix') else config.wavelength
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        rest_wave = config.wavelength
        mid_wave = config.mid_wavelength
        disp_scale = config.dispersion_scale
        
    vel_pix_0 = (rest_wave - mid_wave) / disp_scale
    width_pix_0 = 0.02888811 / disp_scale

    vel0 = vel_pix_0 * np.ones_like(int0)
    width0 = width_pix_0 * np.ones_like(int0)
    
    tasks = []
    for i in range(bb):
        x0 = np.stack((int0[:,i], vel0[:,i], width0[:,i]), axis=0).flatten()
        tasks.append((
            x0, meas[:,:,[i]], mask[:,[i]], lam_i, lam_v, lam_w,
            imager.pixelated, imager.spectral_orders, OPTIMIZER, maxiter, DATA_FIDELITY
        ))

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_worker_scipy_col)(*t) for t in tasks
    )
    
    rec = np.zeros((3,aa,bb))
    for i, res_x in enumerate(results):
        rec[:,:,i] = res_x.reshape(3,aa)

    losses = []
    return rec, losses

def _worker_scipy_col2(x0, meas_slice, mask_slice, lam_i, lam_v, lam_w, 
                       pixelated, spectral_orders, OPTIMIZER, maxiter, DATA_FIDELITY,
                       bg_proj_matrix, ratio_i2, ratio_bkg):

    def obj_ls_local(x, meas, mask, lam_i, lam_v, lam_w):
        aa, bb = meas.shape[1:]
        
        int1, vel1, wid1, int2, vel2, wid2, bkg = np.reshape(x, (7,aa,bb))
            
        proj1 = forward_op(int1, vel1, wid1,
            pixelated=pixelated, mask=mask,
            spectral_orders=spectral_orders)
            
        proj2 = forward_op(int2, vel2, wid2,
            pixelated=pixelated, mask=mask,
            spectral_orders=spectral_orders)
        
        # Mask the background parameter before projecting to respect valid spatial bounds
        proj_bg = np.einsum('oij,jk->oik', bg_proj_matrix, bkg * mask)
        
        diff = (proj1 + proj2 + proj_bg) - meas
        
        regu = (
            lam_v*np.sum(np.diff(vel1, axis=0)**2) + 
            lam_w*np.sum(np.diff(wid1, axis=0)**2) +
            lam_i*np.sum(np.diff(int1, axis=0)**2) +
            
            lam_v*np.sum(np.diff(vel2, axis=0)**2) + 
            lam_w*np.sum(np.diff(wid2, axis=0)**2) +
            (lam_i * ratio_i2)*np.sum(np.diff(int2, axis=0)**2) +
            
            (lam_i * ratio_bkg)*np.sum(np.diff(bkg, axis=0)**2)
        )

        if DATA_FIDELITY == 'L2':
            return np.sum(diff**2) + regu
        elif DATA_FIDELITY == 'L1':
            return np.sum(abs(diff)) + regu

    res = minimize(
        obj_ls_local, 
        x0, 
        args=(meas_slice, mask_slice, lam_i, lam_v, lam_w), 
        method=OPTIMIZER,
        options={'disp':False, 'maxiter':maxiter}
    )
    return res.x

def scipy_solver_parallel2(
    imager=None,
    OPTIMIZER='L-BFGS-B',
    DATA_FIDELITY='L2',
    lam_i=5e2,
    lam_v=5e2,
    lam_w=1e0,
    maxiter=10000,
    n_jobs=-1,
    frac1=0.8620,
    frac2=0.0521,
    frac_bg=0.0860,
    cent1=195.11723,
    wid1=0.02981,
    cent2=195.17723,
    wid2=0.02981,
    bg_shape_norm=[0.04762] * 21,
    return_full=False
    ):

    meas = imager.meas3dar.copy()
    mask = imager.mask
    if mask is None:
        mask = np.ones_like(meas[0])
    aa, bb = meas[0].shape

    rest_wave = imager.srpix.rest_wavelength if hasattr(imager, 'srpix') else config.wavelength
    mid_wave = imager.mid_wavelength if imager is not None else config.mid_wavelength
    disp_scale = imager.dispersion_scale if imager is not None else config.dispersion_scale
    
    if bg_shape_norm is None:
        # Fallback to a flat continuum if no optimal shape is provided
        bg_shape_norm = np.ones(21) / 21.0
    else:
        bg_shape_norm = np.array(bg_shape_norm)
    
    I = np.eye(aa)
    bg_cube = bg_shape_norm[:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
    bg_proj_matrix = forward_op_tomo_3d(bg_cube, orders=imager.spectral_orders)
    
    # Scale intensity regularizations dynamically to keep optimization well-conditioned
    ratio_i2 = (frac1 / max(frac2, 1e-6))**2
    ratio_bkg = (frac1 / max(frac_bg, 1e-6))**2 * 100.0 # Extra strong spatial lock for bkg
    
    int0 = meas[0].copy()
    int1_0 = int0 * frac1
    int2_0 = int0 * frac2
    bkg_0  = int0 * frac_bg
    
    vel1_pix_0 = (cent1 - mid_wave) / disp_scale
    wid1_pix_0 = wid1 / disp_scale
    
    vel2_pix_0 = (cent2 - mid_wave) / disp_scale
    wid2_pix_0 = wid2 / disp_scale
    
    v1_0 = vel1_pix_0 * np.ones_like(int0)
    w1_0 = wid1_pix_0 * np.ones_like(int0)
    
    v2_0 = vel2_pix_0 * np.ones_like(int0)
    w2_0 = wid2_pix_0 * np.ones_like(int0)

    tasks = []
    for i in range(bb):
        x0 = np.stack((int1_0[:,i], v1_0[:,i], w1_0[:,i],
                       int2_0[:,i], v2_0[:,i], w2_0[:,i],
                       bkg_0[:,i]), axis=0).flatten()
        tasks.append((
            x0, meas[:,:,[i]], mask[:,[i]], lam_i, lam_v, lam_w,
            imager.pixelated, imager.spectral_orders, OPTIMIZER, maxiter, DATA_FIDELITY,
            bg_proj_matrix, ratio_i2, ratio_bkg
        ))

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_worker_scipy_col2)(*t) for t in tasks
    )
    
    n_out = 7 if return_full else 3
    rec = np.zeros((n_out, aa, bb))
    for i, res_x in enumerate(results):
        rec[:, :, i] = res_x.reshape(7, aa)[:n_out, :]

    losses = []
    return rec, losses

def diffusion_solver(
    imager=None,
    model_path='model-10.pt',
    grad_scale=[1,1,1],
    num_samples=5
):
    meas = imager.meas3dar.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    channels = len(imager.spectral_orders)
    model = Unet(
        # channels=3,
        channels = channels,
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    ).to(device)
    data = torch.load(str(config.diffusion_model_dir)+'/'+model_path, map_location=device, weights_only=True)
    adapted_dict = {k[6:]: v for k, v in data['model'].items() if k.startswith('model.')}
    model.load_state_dict(adapted_dict)
    model.eval()

    def forward_op(x, device=None):
        return forward_op_torch(
        true_intensity=x[:,0], 
        true_doppler=x[:,1], 
        true_linewidth=x[:,2], 
        device=device
    )

    # Initialize the diffusion process
    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 1000,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        recon = True,
        measurement=torch.tensor(meas).to(device),
        beta_schedule='cosine',
        grad_scale=torch.tensor(grad_scale).to(device),
        forward_op=forward_op,
        device=device,
        mode='all'
    )

    # Generate new samples
    samples, norms, grad_norms, rmses = diffusion.sample(batch_size=num_samples)

    recon = samples.mean(dim=0).detach().cpu().numpy()

    return recon, norms

''' Encapsulated the *comparison_testbed_multi.py* into a function. '''
def comparison_test_multi(path_data, data, savepath, single_param4dar=False, save=False, numdetectors=3, dbsnr=50, 
                          noise_model='poisson', solver='scipy', **kwargs):
    if single_param4dar:
        param4dar = np.load(path_data+data)[[0]]
    else:
        param4dar = np.load(path_data+data)
    if len(param4dar.shape)<4:
        param4dar = param4dar[np.newaxis]

    M = param4dar.shape[-1]

    def inf_priorer(param4dar):
        means = []
        for i in range(len(param4dar)):
            means.append(datacube_generator(param4dar[i]).mean(axis=(1,2)))
        return np.array(means).mean(axis=0)
        
    mask = np.array([[(i + j) % 2 for j in range(M)] for i in range(M)])
    mask= np.ones_like(mask)
    # print('Mask1')
    Imgr = Imager(pixelated=True, mask=mask, dbsnr=dbsnr, max_count=dbsnr**2/0.9, noise_model=noise_model, 
    # Imgr = Imager(pixelated=True, mask=mask, dbsnr=dbsnr, noise_model='gaussian', 
        spectral_orders=[0,-1,1,-2,2][:numdetectors])

    # sr = Source(param3d=param4dar[0],pix=True)
    # m=Imgr.get_measurements(sources=sr)
    # Imgr.plot('poisson_np: {}'.format(poisson_sn))

    # SCIPY
    if solver == 'scipy':
        Rec = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            pix=True,
            solver=scipy_solver,
            **kwargs
        )
    # Tomoinv
    elif solver == 'tomoinv':
        Rec = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            pix=True,
            solver=tomoinv,
            **kwargs
        )
    # 2D GD
    elif solver == 'gd':
        Rec = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            pix=True,
            solver=grad_descent_solver,
            **kwargs
        )
    # U-Net
    elif solver == 'nn':
        Rec = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            pix=True,
            solver=nn_solver,
            **kwargs
        )
    # Diffusion DPS
    elif solver == 'diffusion':
        Rec = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            pix=True,
            solver=diffusion_solver,
            **kwargs
        )
    # SMART
    elif solver == 'smart':
        Rec = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            pix=True,
            solver=smart,
            **kwargs
        )
    # Invalid Solver Type
    else:
        return "Invalid Solver Type!"

    recons = Rec.solve(num_realizations=1)
    Rec.recons[0].plot_loss()
    fig, ax = recons[0].plot(compare=True, title=f'{Rec.solver.__name__}')
    print('mask: {}'.format(mask[:2,:2]))
    print('Solver: {}'.format(Rec.solver.__name__))
    print('Solver Params: {}'.format(Rec.solver_params))
    print('Recon Time Avg: {:.2f} s'.format(Rec.times.mean()))
    print('RMSE_phy Avg (per Img): {}'.format(Rec.rmse_phy.mean(axis=1)))
    print('RMSE_phy Avg: {}'.format(Rec.rmse_phy.mean(axis=(0,1))))
    print('MAE_phy Avg: {}'.format(Rec.mae_phy.mean(axis=(0,1))))
    print('Bias_phy Avg (per Img): {}'.format(Rec.bias_phy.mean(axis=1)))
    print('Bias_phy Avg: {}'.format(Rec.bias_phy.mean(axis=(0,1))))
    print('RMSE_pix Avg (per Img): {}'.format(Rec.rmse_pix.mean(axis=1)))
    print('RMSE_pix Avg: {}'.format(Rec.rmse_pix.mean(axis=(0,1))))
    # print('SSIMs: {}'.format(recons.ssim))
    # print('SSIM Avg: {}'.format(recons.ssim.mean(axis=0)))

    if save==True:
        now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        name = (f'{now}_{Rec.solver.__name__}_{data[:-4]}_K_{numdetectors}_{noise_model}_dbsnr_{dbsnr}')
        savedir = savepath+name
        os.mkdir(savedir)
        recon_summary = [
        '############## Recon Parameters ############## \n',
        'mask: {} \n'.format(mask[:2,:2]),
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
    
    return Rec # May return plots and numerical results if required.